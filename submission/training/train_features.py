import copy
import json
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from model import Head
from splits import ROOT

def fit(x,y,train,val,config,device,epochs,output,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    k=int(y.max()+1);batch=256
    model=Head(x.shape[1],k,config).to(device)
    # Marginal feature statistics are computed only on training rows.
    total=np.zeros(x.shape[1],dtype=np.float64);sq=total.copy()
    for sl in range(0,len(train),512):
        v=x[train[sl:sl+512]].astype(np.float64)
        total+=v.sum(0);sq+=(v*v).sum(0)
    mean=total/len(train)
    inv=np.minimum(100.,1/np.sqrt(np.maximum(sq/len(train)-mean**2,1e-6)))
    model.mean.copy_(torch.tensor(mean));model.scale.copy_(torch.tensor(inv))
    opt=torch.optim.AdamW(model.parameters(),lr=0.0007 if config=='mlp' else 0.0005,weight_decay=0.025)
    best=-1;rows=[]
    counts=np.bincount(y[train],minlength=k);weights=torch.tensor(len(train)/(k*counts),dtype=torch.float32,device=device)
    if not val.size:best_epoch=epochs
    for epoch in range(1,epochs+1):
        model.train()
        order=rng.permutation(train)
        for ids in np.array_split(order,max(1,(len(order)+batch-1)//batch)):
            v=torch.from_numpy(x[ids]).to(device);t=torch.from_numpy(y[ids]).to(device)
            pred=model(v);loss=torch.nn.functional.cross_entropy(pred,t,weight=weights)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if val.size:
            model.eval();probs=[]
            with torch.no_grad():
                for sl in range(0,len(val),batch*2):
                    probs.append(model(torch.from_numpy(x[val[sl:sl+batch*2]]).to(device)).cpu().numpy())
            logits=np.concatenate(probs);score=balanced_accuracy_score(y[val],logits.argmax(1))
            rows.append(float(score))
            if score>best:
                best=float(score);best_epoch=epoch
                best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()});best_logits=logits
        else:
            best_state=None
    output.mkdir(parents=True,exist_ok=True)
    state=best_state if val.size else model.cpu().state_dict()
    torch.save(dict(kind=config,dim=x.shape[1],state=state),output/'model.pt')
    if val.size:np.savez_compressed(output/'val.npz',ids=val,logits=best_logits)
    tmp=output/'metrics.tmp';tmp.write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=rows,train=len(train),val=len(val))));os.replace(tmp,output/'metrics.json')
    return best

def development(task,device):
    base=ROOT/'cache'/task
    x=np.load(base/'features-v1.npy',mmap_mode='r');y=np.load(base/'y.npy');g=np.load(base/'groups.npy')
    plan=json.loads((base/'splits.json').read_text())
    dev=~np.isin(g,plan['blind'])
    for fold,subjects in enumerate(plan['dev_folds']):
        mask=np.isin(g,subjects);tr=np.flatnonzero(dev&~mask);va=np.flatnonzero(mask)
        for kind in ['linear','mlp']:
            out=ROOT/'runs'/'development'/task/kind/f'fold{fold}'
            if (out/'metrics.json').is_file():continue
            score=fit(x,y,tr,va,kind,device,16,out,26000+fold)
            print(task,kind,fold,round(score,4),flush=True)
