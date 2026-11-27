import json,sys
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from geometry import Geometry
from neural import compact_public
from splits import ROOT

def fit_geometry(task,tr,va,out,device='cuda',epochs=1,seed=169,kind='geometry'):
    b=ROOT/'cache'/task;info=json.loads((b/'info.json').read_text())
    x=np.load(b/'x.npy',mmap_mode='r');m=np.load(b/'mask.npy',mmap_mode='r');y=np.load(b/'y.npy');g=np.load(b/'groups.npy')
    pub=np.load(b/'foundation.npy',mmap_mode='r');N=len(info['label_values'])
    from temporal import Temporal
    model_type=Temporal if kind=='temporal' else Geometry
    model=model_type(info).to(device).fit(x,m,y,g,tr,device,seed)
    def feats(ids):
        a=[]
        with torch.inference_mode():
            for j in range(0,len(ids),80):
                ix=ids[j:j+80];v=model(torch.as_tensor(x[ix],device=device),torch.as_tensor(m[ix],device=device))
                h=compact_public(torch.tensor(pub[ix],device=device,dtype=torch.float32),info)
                a.append(torch.cat([v,h],1).cpu())
        return torch.cat(a)
    # Balance classes and moderately rebalance subjects within each class.
    wt=np.zeros(len(tr))
    for k in range(N):
        pos=np.flatnonzero(y[tr]==k);_,i,n=np.unique(g[tr[pos]],return_inverse=True,return_counts=True)
        w=n[i]**-.6;wt[pos]=w/w.sum()
    w=torch.as_tensor(wt/wt.sum(),dtype=torch.float64,device=device)
    v=feats(tr).to(device,dtype=torch.float64)
    mu=w@v;v-=mu
    inv=(w@v.square()).clamp_min(.0001).rsqrt();v*=inv
    v=v.clamp(-8,8)
    t=torch.nn.functional.one_hot(torch.tensor(y[tr],device=device),N).double()
    gram=v.T@(w[:,None]*v)
    rhs=v.T@(w[:,None]*(t-1/N))
    # A fixed ridge penalty avoids reselecting a regularizer for each validation fold.
    gram.diagonal().add_(2.0)
    coef=torch.linalg.solve(gram,rhs)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind=kind,state=model.state_dict(),mean=mu.float(),inv=inv.float(),coef=coef.float()),out/'model.pt')
    score=0
    if len(va):
        v=feats(va).to(device)
        z=(((v-mu.float())*inv.float()).clamp(-8,8)@coef.float()).cpu().numpy()
        score=float(balanced_accuracy_score(y[va],z.argmax(1)))
        np.savez_compressed(out/'val.npz',ids=va,logits=z)
    (out/'metrics.json').write_text(json.dumps(dict(best=score,epoch=1,curve=[score])))
    return score

if __name__=='__main__':
    task=sys.argv[1];base=ROOT/'cache'/task;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.22)
    g=np.load(base/'groups.npy');sp=json.loads((base/'splits.json').read_text())
    for f,groups in enumerate(sp['dev_folds']):
        out=ROOT/'runs/development'/task/'geometry'/f'fold{f}'
        if (out/'metrics.json').is_file():continue
        v=np.isin(g,groups);tr=np.flatnonzero(~np.isin(g,sp['blind'])&~v);va=np.flatnonzero(v)
        print(task,f,fit_geometry(task,tr,va,out),flush=True)
