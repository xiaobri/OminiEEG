import json,math,sys
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from splits import ROOT
from full import Full

def fit_full(task,tr,va,out,device='cuda',epochs=19,seed=97,model_class=Full):
    b=ROOT/'cache'/task;info=json.loads((b/'info.json').read_text())
    x=np.load(b/'x.npy',mmap_mode='r');m=np.load(b/'mask.npy',mmap_mode='r');y=np.load(b/'y.npy');g=np.load(b/'groups.npy')
    torch.manual_seed(seed);rng=np.random.default_rng(seed);net=model_class(info,ROOT/'assets/cbramod-pretrained.pth').to(device)
    if hasattr(net,'fit_source'):
        net.fit_source(x,m,g,tr,device)
    opt=torch.optim.AdamW([dict(params=net.backbone.encoder.patch_embedding.parameters(),lr=5e-5),
            dict(params=net.adapters.parameters(),lr=3e-4),dict(params=list(net.head.parameters())+list(net.read.parameters()),lr=8e-4)],weight_decay=.03)
    wt=np.zeros(len(tr))
    for k in np.unique(y[tr]):
        ix=np.flatnonzero(y[tr]==k);_,inv,n=np.unique(g[tr[ix]],return_inverse=True,return_counts=True)
        a=n[inv]**-.5;wt[ix]=a/a.sum()
    wt/=wt.sum();train_n=min(len(tr),4200)
    batch=24 if x.shape[1]*min(x.shape[-1],6000)>30000 else 40
    best=-1;curve=[];out.mkdir(parents=True,exist_ok=True)
    def batch_forward(ix):
        sig=torch.as_tensor(x[ix],device=device);mask=torch.as_tensor(m[ix],device=device)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):z=net(sig,mask)
        return z.float()
    for e in range(1,epochs+1):
        net.train()
        # Source checkpoint's own dropout stays disabled in frozen transformer blocks.
        net.backbone.encoder.encoder.eval()
        frac=min(1.,e/2)*(.15+.85*.5*(1+math.cos(math.pi*(e-1)/19)))
        for group,lr in zip(opt.param_groups,[5e-5,3e-4,8e-4]):group['lr']=lr*frac
        order=rng.choice(tr,train_n,replace=True,p=wt)
        for ix in np.array_split(order,math.ceil(len(order)/batch)):
            z=batch_forward(ix);loss=torch.nn.functional.cross_entropy(z,torch.as_tensor(y[ix],device=device),label_smoothing=.04)
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if len(va):
            with torch.inference_mode():
                net.eval()
                logits=np.concatenate([batch_forward(va[j:j+batch]).cpu().numpy() for j in range(0,len(va),batch)])
            score=float(balanced_accuracy_score(y[va],logits.argmax(1)));curve.append(score)
        else: score=0.
        if (len(va) and score>best) or (not len(va) and e==epochs):
            best=score;best_epoch=e;torch.save(dict(kind=getattr(net,'checkpoint_kind','full'),delta=net.delta()),out/'model.pt')
            if len(va):np.savez_compressed(out/'val.npz',ids=va,logits=logits)
        if e%5==0:print(task,out.name,e,round(best,4),flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=curve)))
    return best

if __name__=='__main__':
    task=sys.argv[1];b=ROOT/'cache'/task;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.mha.set_fastpath_enabled(False)
    g=np.load(b/'groups.npy');sp=json.loads((b/'splits.json').read_text())
    for f,groups in enumerate(sp['dev_folds']):
        out=ROOT/'runs/development'/task/'full'/f'fold{f}'
        if (out/'metrics.json').is_file():continue
        v=np.isin(g,groups);tr=np.flatnonzero(~np.isin(g,sp['blind'])&~v);va=np.flatnonzero(v)
        print(task,f,fit_full(task,tr,va,out,seed=97+f),flush=True)
