import json
import math
import os
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from spectrogram import Spectrogram, SpectrumHead
from splits import ROOT
from train_neural import compact_array


def spectra(base, info, device):
    path = base/'spectrogram.npy'
    if not path.is_file():
        extractor = Spectrogram(info).eval().to(device)
        x = np.load(base/'x.npy', mmap_mode='r')
        mask = np.load(base/'mask.npy', mmap_mode='r')
        tmp = path.with_name(f'spectrogram.{os.getpid()}.tmp')
        dest = None
        with torch.inference_mode():
            for i in range(0,len(x),64):
                v = extractor(torch.as_tensor(x[i:i+64].copy(),device=device),torch.as_tensor(mask[i:i+64].copy(),device=device)).cpu().numpy()
                if not np.isfinite(v).all():
                    raise ValueError('nonfinite spectrum')
                if dest is None:
                    dest = np.lib.format.open_memmap(tmp,mode='w+',dtype='float16',shape=(len(x),*v.shape[1:]))
                dest[i:i+len(v)] = v
        dest.flush()
        del dest
        os.replace(tmp,path)
    return np.load(path,mmap_mode='r')


def fit_spectrum(task,tr,va,out,device='cuda',epochs=25,seed=321,style=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    base = ROOT/'cache'/task
    info = json.loads((base/'info.json').read_text())
    x = spectra(base,info,device)
    pub = compact_array(base,info)
    y = np.load(base/'y.npy')
    groups = np.load(base/'groups.npy')
    model = SpectrumHead(info,pub.shape[1],x.shape[2]).to(device)
    ids = rng.choice(tr,min(len(tr),1024),replace=False)
    proj = pub[ids]
    model.pm.copy_(torch.as_tensor(proj.mean(0)))
    model.ps.copy_(torch.as_tensor(1/np.maximum(.02,proj.std(0))))
    example = x[ids].astype('float32')
    model.mean.copy_(torch.as_tensor(example.mean((0,3),keepdims=True)))
    model.scale.copy_(torch.as_tensor(1/np.maximum(.15,example.std((0,3),keepdims=True))))
    del example
    opt = torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.035)
    weights = np.zeros(len(tr))
    for k in np.unique(y[tr]):
        rows = np.flatnonzero(y[tr]==k)
        _,ii,counts = np.unique(groups[tr[rows]],return_inverse=True,return_counts=True)
        w = counts[ii]**-.6
        weights[rows] = w / w.sum()
    weights /= weights.sum()
    size = min(len(tr),12300)
    out.mkdir(parents=True,exist_ok=True)
    best = -1
    curve = []
    if style:
        source_groups=np.unique(groups[tr])
        profile=[]
        embeddings=[]
        group_class=[]
        for group in source_groups:
            ix=tr[groups[tr]==group]
            if len(np.unique(y[ix]))!=1:
                raise ValueError('Subject label changes in the source domain regularizer')
            profile.append(x[ix].astype('float32').mean((0,3),keepdims=False))
            embeddings.append(pub[ix].mean(0))
            group_class.append(y[ix[0]])
        profile=np.stack(profile)
        embeddings=np.stack(embeddings)
        for label in np.unique(group_class):
            same=np.asarray(group_class)==label
            profile[same]-=profile[same].mean(0)
            embeddings[same]-=embeddings[same].mean(0)
        profile=torch.as_tensor(profile,device=device)
        embeddings=torch.as_tensor(embeddings,device=device)
    def compute(ix):
        local=torch.as_tensor(x[ix],device=device).float()
        public=torch.as_tensor(pub[ix],device=device)
        if style and model.training:
            current=np.searchsorted(source_groups,groups[ix])
            donor=rng.integers(len(source_groups),size=len(ix))
            strength=torch.as_tensor(rng.uniform(0.,1.,size=len(ix)),device=device,dtype=torch.float32)
            local=local+(profile[donor]-profile[current]).unsqueeze(-1)*strength[:,None,None,None]
            public=public+(embeddings[donor]-embeddings[current])*strength[:,None]
        return model(local,public)
    for epoch in range(1,epochs+1):
        model.train()
        opt.param_groups[0]['lr'] = .0009*(.18+.82*.5*(1+math.cos(math.pi*(epoch-1)/25)))
        rows = rng.choice(tr,size,replace=True,p=weights)
        for ix in np.array_split(rows,(size+95)//96):
            pred = compute(ix)
            loss = torch.nn.functional.cross_entropy(pred,torch.as_tensor(y[ix],device=device),label_smoothing=.04)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
        if len(va):
            model.eval()
            with torch.inference_mode():
                logits = np.concatenate([compute(va[i:i+96]).cpu().numpy() for i in range(0,len(va),96)])
            score = float(balanced_accuracy_score(y[va],logits.argmax(1)))
            curve.append(score)
        else:
            score = 0.
        if (len(va) and score>best) or (not len(va) and epoch==epochs):
            best = score
            best_epoch = epoch
            torch.save(dict(kind='style' if style else 'spectrum',dim=pub.shape[1],bands=x.shape[2],state=model.state_dict()),out/'model.pt')
            if len(va):
                np.savez_compressed(out/'val.npz',ids=va,logits=logits)
    (out/'metrics.json').write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=curve)))
    return best


if __name__=='__main__':
    task=sys.argv[1]
    base=ROOT/'cache'/task
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20)
    g=np.load(base/'groups.npy')
    sp=json.loads((base/'splits.json').read_text())
    for f,subjects in enumerate(sp['dev_folds']):
        held=np.isin(g,subjects)
        out=ROOT/'runs/development'/task/'spectrum'/f'fold{f}'
        if not (out/'metrics.json').is_file():
            print(task,f,fit_spectrum(task,np.flatnonzero(~np.isin(g,sp['blind'])&~held),np.flatnonzero(held),out,seed=321+f),flush=True)
