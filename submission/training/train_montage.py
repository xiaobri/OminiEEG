import json
import math
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from montage import Montage
from splits import ROOT


def fit_montage(task,tr,va,out,device='cuda',epochs=24,seed=174):
    base=ROOT/'cache'/task
    info=json.loads((base/'info.json').read_text())
    raw=np.load(base/'x.npy',mmap_mode='r')
    mask=np.load(base/'mask.npy',mmap_mode='r')
    labels=np.load(base/'y.npy')
    groups=np.load(base/'groups.npy')
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=Montage(info,ROOT/'assets/cbramod-pretrained.pth').to(device)
    rates=[.0016,3e-5,4e-5,.0007]
    opt=torch.optim.AdamW([
        dict(params=[net.offset,net.gain],lr=rates[0]),
        dict(params=net.backbone.encoder.patch_embedding.parameters(),lr=rates[1]),
        dict(params=net.backbone.encoder.encoder.layers[-4:].parameters(),lr=rates[2]),
        dict(params=[*net.read.parameters(),*net.head.parameters()],lr=rates[3])],weight_decay=.025)
    weights=np.zeros(len(tr))
    for k in np.unique(labels[tr]):
        ix=np.flatnonzero(labels[tr]==k)
        _,idx,counts=np.unique(groups[tr[ix]],return_inverse=True,return_counts=True)
        w=counts[idx]**-.55
        weights[ix]=w/w.sum()
    weights/=weights.sum()
    size=min(len(tr),7500)
    batch=32 if raw.shape[-1]>=1600 else 48
    def forward(ix):
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            z=net(torch.as_tensor(raw[ix],device=device),torch.as_tensor(mask[ix],device=device))
        return z.float()
    out.mkdir(parents=True,exist_ok=True)
    curve=[]
    best=-1.
    for epoch in range(1,epochs+1):
        net.train()
        for layer in net.backbone.encoder.encoder.layers[:-4]:
            layer.eval()
        frac=min(1.,epoch/3)*(.15+.85*.5*(1+math.cos(math.pi*(epoch-1)/24)))
        for pg,rate in zip(opt.param_groups,rates):
            pg['lr']=rate*frac
        rows=rng.choice(tr,size,replace=True,p=weights)
        for ix in np.array_split(rows,(len(rows)+batch-1)//batch):
            z=forward(ix)
            loss=torch.nn.functional.cross_entropy(z,torch.as_tensor(labels[ix],device=device),label_smoothing=.06)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.5)
            opt.step()
        if len(va):
            net.eval()
            with torch.inference_mode():
                logits=np.concatenate([forward(va[i:i+batch]).cpu().numpy() for i in range(0,len(va),batch)])
            score=float(balanced_accuracy_score(labels[va],logits.argmax(1)))
            curve.append(score)
        else:
            score=0.
        if (len(va) and score>best) or (not len(va) and epoch==epochs):
            best=score
            best_epoch=epoch
            torch.save(dict(kind='montage',delta=net.delta()),out/'model.pt')
            if len(va):
                np.savez_compressed(out/'val.npz',ids=va,logits=logits)
        if epoch%6==0:
            print(task,out.name,epoch,round(best,4),flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=curve)))
    return best


if __name__=='__main__':
    task=sys.argv[1]
    base=ROOT/'cache'/task
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.21)
    torch.backends.mha.set_fastpath_enabled(False)
    g=np.load(base/'groups.npy')
    sp=json.loads((base/'splits.json').read_text())
    for f,sub in enumerate(sp['dev_folds']):
        v=np.isin(g,sub)
        out=ROOT/'runs/development'/task/'montage'/f'fold{f}'
        if not (out/'metrics.json').is_file():
            print(task,f,fit_montage(task,np.flatnonzero(~np.isin(g,sp['blind'])&~v),np.flatnonzero(v),out,seed=174+f),flush=True)
