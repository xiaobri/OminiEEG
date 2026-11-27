import json,os,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from splits import ROOT
from train_neural import compact_array
from foundation import MatrixPair
from tree import encode_booster,evaluate
sys.path.insert(0,str(ROOT/'train_deps'))
import lightgbm as lgb

def select(v,y,g,ids,seed):
    rng=np.random.default_rng(seed)
    ids=rng.choice(ids,min(len(ids),14000),replace=False)
    k=y.max()+1;d=v.shape[1];sums=np.zeros((k,d));sq=np.zeros(d);n=np.zeros(k);features=[]
    # Rank features by class effects, giving each training subject comparable weight within class.
    for label in range(k):
        rows=ids[y[ids]==label]
        subjects=np.unique(g[rows]);n[label]=len(subjects)
        for sub in subjects:
            block=v[rows[g[rows]==sub]]
            sums[label]+=np.mean(block,0,dtype=np.float64)
            sq+=np.einsum('ij,ij->j',block,block,dtype='float64')/(len(block)*len(subjects)*k)
    means=sums/n[:,None];grand=means.mean(0)
    scores=means.var(0)/np.maximum(sq-grand**2,1e-8)
    # Diversify among physics and deep features.
    dim_a=v.a.shape[1];part_sizes=[(0,dim_a,400),(dim_a,d,min(260,d-dim_a))]
    for start,end,keep in part_sizes:
        order=np.argsort(scores[start:end])[::-1][:keep];features.extend((start+order).tolist())
    return np.asarray(features,dtype=np.int64)

def fit_tree(task,tr,va,out,device=None,epochs=220,seed=51):
    base=ROOT/'cache'/task;info=json.loads((base/'info.json').read_text());y=np.load(base/'y.npy');g=np.load(base/'groups.npy')
    v=MatrixPair(np.load(base/'features-v1.npy',mmap_mode='r'),compact_array(base,info))
    cols=select(v,y,g,tr,seed)
    def data(ids):
        return np.concatenate([v[ix][:,cols] for ix in np.array_split(ids,max(1,(len(ids)+511)//512))])
    xt=data(tr);k=len(info['label_values'])
    xtest=data(va) if len(va) else None
    w=np.zeros(len(tr))
    for label in range(k):
        ix=np.flatnonzero(y[tr]==label);_,ii,n=np.unique(g[tr[ix]],return_counts=True,return_inverse=True)
        a=n[ii]**-.35;w[ix]=a/a.sum()
    w*=len(tr)/w.sum()
    params=dict(objective='multiclass' if k>2 else 'binary',verbosity=-1,seed=seed,num_threads=3,
                learning_rate=.055,feature_fraction=.63,bagging_fraction=.82,bagging_freq=1,num_leaves=19,
                max_depth=6,min_data_in_leaf=max(12,min(100,len(tr)//250)),lambda_l2=6.,max_bin=95,force_col_wise=True)
    if k>2:params['num_class']=k
    data_train=lgb.Dataset(xt,label=y[tr],weight=w,free_raw_data=False);booster=lgb.train(params,data_train,num_boost_round=epochs)
    if len(va):
        stages=list(range(15,epochs+1,15))
        if not stages or stages[-1]!=epochs:stages.append(epochs)
        best=-1;curve=[]
        for n in stages:
            z=booster.predict(xtest,num_iteration=n,raw_score=True,num_threads=3)
            if k==2:z=np.column_stack((np.zeros(len(z)),z))
            score=float(balanced_accuracy_score(y[va],z.argmax(1)));curve.append(score)
            if score>best:best=score;best_epoch=n;ref=z
    else:best=0.;best_epoch=epochs;curve=[]
    bst=lgb.Booster(model_str=booster.model_to_string(num_iteration=best_epoch))
    tree=encode_booster(bst)
    sample=xt[:50];z=bst.predict(sample,raw_score=True,num_threads=3)
    if k==2:z=np.column_stack((np.zeros(len(z)),z))
    np.testing.assert_allclose(evaluate(tree,sample),z,atol=1e-4,rtol=1e-4)
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind='tree',cols=torch.from_numpy(cols),tree={key:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for key,v in tree.items()}),out/'model.pt')
    if len(va):np.savez_compressed(out/'val.npz',ids=va,logits=ref.astype('float32'))
    (out/'metrics.json').write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=curve)))
    return best

if __name__=='__main__':
    task=sys.argv[1];base=ROOT/'cache'/task;g=np.load(base/'groups.npy');sp=json.loads((base/'splits.json').read_text())
    for f,groups in enumerate(sp['dev_folds']):
        out=ROOT/'runs/development'/task/'tree'/f'fold{f}'
        if (out/'metrics.json').is_file():continue
        t=np.isin(g,groups);tr=np.flatnonzero(~np.isin(g,sp['blind'])&~t);va=np.flatnonzero(t)
        print(task,f,fit_tree(task,tr,va,out,epochs=180,seed=51+f),flush=True)
