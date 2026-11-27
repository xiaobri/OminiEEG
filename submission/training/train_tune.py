import json,math,os,sys
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from splits import ROOT
from tune import Tuned

def build_hidden(task,model,device):
    b=ROOT/'cache'/task;path=b/'hidden10.npy'
    if not path.exists():
        model.eval();x=np.load(b/'x.npy',mmap_mode='r');m=np.load(b/'mask.npy',mmap_mode='r');dest=None
        with torch.no_grad():
            for at in range(0,len(x),32):
                h=model.frozen(torch.as_tensor(x[at:at+32].copy(),device=device),torch.as_tensor(m[at:at+32].copy(),device=device)).cpu().numpy()
                if not np.isfinite(h).all():raise ValueError('nonfinite intermediate tokens')
                if dest is None:dest=np.lib.format.open_memmap(path.with_suffix('.tmp'),mode='w+',dtype='float16',shape=(len(x),*h.shape[1:]))
                dest[at:at+len(h)]=h
        dest.flush();del dest;os.replace(path.with_suffix('.tmp'),path)
    return np.load(path,mmap_mode='r')

def fit_tune(task,tr,va,out,device='cuda',epochs=20,seed=89):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);b=ROOT/'cache'/task
    info=json.loads((b/'info.json').read_text());y=np.load(b/'y.npy');g=np.load(b/'groups.npy')
    model=Tuned(info,ROOT/'assets/cbramod-pretrained.pth').to(device)
    x=build_hidden(task,model,device)
    opt=torch.optim.AdamW([{'params':model.base.encoder.encoder.layers[-2:].parameters(),'lr':7e-5},
                           {'params':list(model.local.parameters())+list(model.readout.parameters()),'lr':8e-4}],weight_decay=.035)
    wt=np.zeros(len(tr))
    for cls in np.unique(y[tr]):
        ix=np.flatnonzero(y[tr]==cls);_,i,n=np.unique(g[tr[ix]],return_inverse=True,return_counts=True)
        w=n[i]**(-.5);wt[ix]=w/w.sum()
    wt/=wt.sum();n=min(len(tr),9216);batch=64;best=-1;curve=[];out.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,epochs+1):
        # Keep schedule tied to the original search horizon, including during refits.
        frac=(1+math.cos(math.pi*(epoch-1)/20))*.4+.2
        for pg,lr in zip(opt.param_groups,[7e-5,8e-4]):pg['lr']=lr*frac
        model.train();order=rng.choice(tr,n,replace=True,p=wt)
        for ix in np.array_split(order,math.ceil(n/batch)):
            h=torch.from_numpy(x[ix]).to(device).float();t=torch.from_numpy(y[ix]).to(device)
            mix=rng.beta(.22,.22) if rng.random()<.5 else 1.
            mixed=h*mix+h.flip(0)*(1-mix) if mix<1. else h
            z=model.classify(mixed)
            loss=mix*torch.nn.functional.cross_entropy(z,t,label_smoothing=.035)+(1-mix)*torch.nn.functional.cross_entropy(z,t.flip(0),label_smoothing=.035)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.5);opt.step()
        if len(va):
            with torch.inference_mode():
                model.eval()
                z=np.concatenate([model.classify(torch.from_numpy(x[va[start:start+batch]]).to(device)).cpu().numpy() for start in range(0,len(va),batch)])
            score=float(balanced_accuracy_score(y[va],z.argmax(1)));curve.append(score)
        else:score=0.
        if (len(va) and score>best) or (not len(va) and epoch==epochs):
            best=score;best_epoch=epoch
            torch.save(dict(kind='tune',delta=model.delta()),out/'model.pt')
            if len(va):np.savez_compressed(out/'val.npz',ids=va,logits=z)
        if epoch%5==0:print(task,out.name,epoch,best,flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=curve)))
    return best

if __name__=='__main__':
    task=sys.argv[1];b=ROOT/'cache'/task;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.mha.set_fastpath_enabled(False)
    g=np.load(b/'groups.npy');sp=json.loads((b/'splits.json').read_text())
    for f,groups in enumerate(sp['dev_folds']):
        out=ROOT/'runs/development'/task/'tune'/f'fold{f}'
        if (out/'metrics.json').is_file():continue
        t=np.isin(g,groups);tr=np.flatnonzero(~np.isin(g,sp['blind'])&~t);va=np.flatnonzero(t)
        print(task,f,fit_tune(task,tr,va,out,seed=89+f),flush=True)
