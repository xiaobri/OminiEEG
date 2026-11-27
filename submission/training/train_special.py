import json,sys
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from steady import Steady,SteadyHead,global_summary
from splits import ROOT,split_groups

def transform(model,ids,raw,mask,device):
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(raw[ix]).to(device),torch.from_numpy(mask[ix]).to(device)).cpu() for ix in np.array_split(ids,max(1,(len(ids)+95)//96))])

def fit_steady(task,tr,va,out,device='cuda',epochs=32,seed=81,kind='steady'):
    base=ROOT/'cache'/task;info=json.loads((base/'info.json').read_text());y=np.load(base/'y.npy');g=np.load(base/'groups.npy')
    raw=np.load(base/'x.npy',mmap_mode='r');mask=np.load(base/'mask.npy',mmap_mode='r');frozen=np.load(base/'foundation.npy',mmap_mode='r')
    from steady_window import SteadyWindow
    from steady_delay import SteadyDelay
    from steady_source import SteadySource
    model_type={'steady':Steady,'window':SteadyWindow,'delay':SteadyDelay,'source':SteadySource}[kind]
    extra={'groups':g} if kind=='source' else {}
    torch.manual_seed(seed)
    def pub(ix):
        return global_summary(torch.from_numpy(frozen[ix].astype('float32')),info).to(device)
    # Cross-fit class templates; a training example never contributes to its own features.
    partitions=split_groups(y,g,tr,3)
    scores=[];targets=[];pubs=[]
    for test in partitions:
        sub=np.setdiff1d(tr,test)
        tmp=model_type(info).to(device).fit(raw,mask,y,sub,device,**extra)
        scores.append(transform(tmp,test,raw,mask,device));targets.extend(y[test]);pubs.append(pub(test))
    proto=model_type(info).to(device).fit(raw,mask,y,tr,device,**extra)
    X=torch.cat(scores).to(device);Y=torch.as_tensor(targets,device=device);G=torch.cat(pubs)
    head=SteadyHead(X.shape[-1],G.shape[1],len(info['label_values'])).to(device)
    head.m.copy_(X.mean((0,1)));head.s.copy_(X.std((0,1)).clamp_min(.01).reciprocal())
    head.gm.copy_(G.mean(0));head.gs.copy_(G.std(0).clamp_min(.05).reciprocal())
    nnopt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.04)
    if len(va):
        xv=transform(proto,va,raw,mask,device).to(device);gv=pub(va)
    best=-1.;best_epoch=epochs;out.mkdir(exist_ok=True,parents=True);curve=[]
    for epoch in range(1,epochs+1):
        head.train()
        for ids in torch.randperm(len(Y),device=device).split(192):
            logits=head(X[ids],G[ids])
            loss=torch.nn.functional.cross_entropy(logits,Y[ids],label_smoothing=.06);nnopt.zero_grad(set_to_none=True);loss.backward();nnopt.step()
        if len(va):
            with torch.no_grad():z=head.eval()(xv,gv).cpu().numpy()
            s=float(balanced_accuracy_score(y[va],z.argmax(1)));curve.append(s)
        else:s=0
        if (len(va) and s>best) or (not len(va) and epoch==epochs):
            best_epoch=epoch;best=s
            torch.save(dict(kind=kind,proto=proto.state_dict(),head=head.state_dict(),dim=X.shape[-1]),out/'model.pt')
            if len(va):np.savez_compressed(out/'val.npz',ids=va,logits=z)
    (out/'metrics.json').write_text(json.dumps(dict(best=best,epoch=best_epoch,curve=curve)))
    return best

if __name__=='__main__':
    task=sys.argv[1];b=ROOT/'cache'/task
    torch.cuda.set_per_process_memory_fraction(.20);torch.set_num_threads(2)
    g=np.load(b/'groups.npy');sp=json.loads((b/'splits.json').read_text())
    for f,groups in enumerate(sp['dev_folds']):
        out=ROOT/'runs/development'/task/'steady'/f'fold{f}'
        if (out/'metrics.json').is_file():continue
        t=np.isin(g,groups);tr=np.flatnonzero(~np.isin(g,sp['blind'])&~t);va=np.flatnonzero(t)
        print(task,f,fit_steady(task,tr,va,out),flush=True)
