import json
import sys

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from motor import Motor
from neural import compact_public
from splits import ROOT


def fit_motor(task,tr,va,out,device='cuda',epochs=1,seed=277):
    base=ROOT/'cache'/task
    info=json.loads((base/'info.json').read_text())
    x=np.load(base/'x.npy',mmap_mode='r')
    m=np.load(base/'mask.npy',mmap_mode='r')
    y=np.load(base/'y.npy')
    g=np.load(base/'groups.npy')
    public=np.load(base/'foundation.npy',mmap_mode='r')
    net=Motor(info).to(device).fit(x,m,y,g,tr,device,seed)
    n=len(info['label_values'])
    def features(ids):
        result=[]
        with torch.no_grad():
            for start in range(0,len(ids),64):
                ix=ids[start:start+64]
                raw=net(torch.as_tensor(x[ix],device=device),torch.as_tensor(m[ix],device=device))
                p=compact_public(torch.as_tensor(public[ix],device=device,dtype=torch.float32),info)
                result.append(torch.cat([raw,p],1))
        return torch.cat(result)
    v=features(tr).double()
    target=torch.nn.functional.one_hot(torch.as_tensor(y[tr],device=device),n).double()-1/n
    weight=np.zeros(len(tr))
    cells=[]
    for k in range(n):
        rows=np.flatnonzero(y[tr]==k)
        subjects=np.unique(g[tr[rows]])
        for sub in subjects:
            cell=rows[g[tr[rows]]==sub]
            weight[cell]=1/(n*len(subjects)*len(cell))
            cells.append((k,cell))
    w=torch.as_tensor(weight,device=device)
    mean=w@v
    v=v-mean
    inv=(w@v.square()).clamp_min(.001).rsqrt()
    v=(v*inv).clamp(-8,8)
    # Penalize score shifts between training subjects within each class.
    mean_class=torch.stack([v[y[tr]==k].mean(0) for k in range(n)])
    shifts=torch.stack([v[cell].mean(0)-mean_class[k] for k,cell in cells])
    gram=v.T@(w[:,None]*v)+(shifts.T@shifts)/len(shifts)
    gram.diagonal().add_(2.)
    coef=torch.linalg.solve(gram,v.T@(w[:,None]*target))
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind='motor',state=net.state_dict(),mean=mean.float(),inv=inv.float(),coef=coef.float()),out/'model.pt')
    score=0.
    if len(va):
        logits=(((features(va)-mean.float())*inv.float()).clamp(-8,8)@coef.float()).cpu().numpy()
        score=float(balanced_accuracy_score(y[va],logits.argmax(1)))
        np.savez_compressed(out/'val.npz',ids=va,logits=logits)
    (out/'metrics.json').write_text(json.dumps(dict(best=score,epoch=1,curve=[score])))
    return score


if __name__=='__main__':
    task=sys.argv[1]
    base=ROOT/'cache'/task
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20)
    g=np.load(base/'groups.npy')
    sp=json.loads((base/'splits.json').read_text())
    for f,sub in enumerate(sp['dev_folds']):
        v=np.isin(g,sub)
        out=ROOT/'runs/development'/task/'motor'/f'fold{f}'
        if not (out/'metrics.json').is_file():
            print(task,f,fit_motor(task,np.flatnonzero(~np.isin(g,sp['blind'])&~v),np.flatnonzero(v),out),flush=True)
