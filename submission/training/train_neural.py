import json,math,os,sys
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from neural import compact_public,EEGAdapter
from splits import ROOT

def compact_array(base,info):
    p=base/'compact.npy'
    if not p.exists():
        source=np.load(base/'foundation.npy',mmap_mode='r')
        tmp=p.with_name(f'compact.{os.getpid()}.tmp')
        dest=None
        for at in range(0,len(source),256):
            v=compact_public(torch.from_numpy(source[at:at+256].astype('float32')),info).numpy()
            if dest is None:dest=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=(len(source),v.shape[1]))
            dest[at:at+len(v)]=v
        dest.flush();del dest;os.replace(tmp,p)
    return np.load(p,mmap_mode='r')

def fit_neural(task,tr,va,out,device='cuda',epochs=22,seed=415):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);b=ROOT/'cache'/task
    info=json.loads((b/'info.json').read_text())
    x=np.load(b/'x.npy',mmap_mode='r');y=np.load(b/'y.npy');g=np.load(b/'groups.npy');m=np.load(b/'mask.npy',mmap_mode='r')
    pub=compact_array(b,info);model=EEGAdapter(info,pub.shape[1]).to(device)
    sel=rng.choice(tr,min(len(tr),1000),replace=False)
    a=x[sel][:,:,::max(1,x.shape[-1]//120)]
    cen=np.median(a,axis=(0,2));disp=np.median(np.abs(a-cen[None,:,None]),axis=(0,2))*1.4826
    model.center.copy_(torch.tensor(cen[None,:,None]));model.inv.copy_(torch.tensor(1/np.maximum(disp,1e-4))[None,:,None])
    v=pub[sel];model.pm.copy_(torch.from_numpy(v.mean(0)));model.ps.copy_(torch.from_numpy(1/np.maximum(v.std(0),.01)))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.012)
    weights=np.zeros(len(tr))
    for k in np.unique(y[tr]):
        ids=np.flatnonzero(y[tr]==k);_,ii,ct=np.unique(g[tr[ids]],return_inverse=True,return_counts=True)
        w=ct[ii]**(-.50);weights[ids]=w/w.sum()
    weights/=weights.sum()
    train_n=min(len(tr),8192 if info['rows']>30000 else 12000);batch=64
    best=-1;curve=[];out.mkdir(parents=True,exist_ok=True)
    def forward(ix):
        xx=torch.from_numpy(x[ix]).to(device);mm=torch.from_numpy(m[ix]).to(device);zz=torch.from_numpy(pub[ix]).to(device)
        return model(xx,mm,zz)
    for ep in range(1,epochs+1):
        model.train()
        for group in opt.param_groups:group['lr']=.001*(.2+.8*min(1.,ep/2)*(.5+.5*math.cos(math.pi*min(ep-1,21)/22)))
        order=rng.choice(tr,train_n,replace=True,p=weights)
        for ix in np.array_split(order,math.ceil(len(order)/batch)):
            z=forward(ix);target=torch.from_numpy(y[ix]).to(device)
            loss=torch.nn.functional.cross_entropy(z,target,label_smoothing=.04);opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if len(va):
            with torch.inference_mode():
                model.eval();z=np.concatenate([forward(va[i:i+batch]).cpu().numpy() for i in range(0,len(va),batch)])
            score=float(balanced_accuracy_score(y[va],z.argmax(1)));curve.append(score)
        else:score=0.
        if (len(va) and score>best) or (not len(va) and ep==epochs):
            best=score;best_epoch=ep;torch.save(dict(kind='neural',dim=pub.shape[1],state=model.state_dict()),out/'model.pt')
            if len(va):np.savez_compressed(out/'val.npz',ids=va,logits=z)
        if ep%5==0:print(task,out.name,ep,'best',round(best,4),flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=curve)))
    return best

if __name__=='__main__':
    task=sys.argv[1];b=ROOT/'cache'/task;torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cudnn.benchmark=False
    g=np.load(b/'groups.npy');sp=json.loads((b/'splits.json').read_text())
    for f,groups in enumerate(sp['dev_folds']):
        out=ROOT/'runs/development'/task/'neural'/f'fold{f}'
        if (out/'metrics.json').is_file():continue
        t=np.isin(g,groups);tr=np.flatnonzero(~np.isin(g,sp['blind'])&~t);va=np.flatnonzero(t)
        print(task,f,fit_neural(task,tr,va,out,seed=415+f),flush=True)
