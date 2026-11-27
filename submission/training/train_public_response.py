"""Train one shared response head on subject-cross-fitted templates and public CBraMod embeddings."""
import json
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from latency_response import LatencyDescriptors
from task_response import TargetDescriptors
from public_response import PublicResponseHead
from splits import ROOT,split_groups
from steady import global_summary
from train_special import transform


def fit_public_response(task,tr,va,out,device='cuda',epochs=12,seed=431,kind='public_target'):
    torch.manual_seed(seed)
    b=ROOT/'cache'/task;info=json.loads((b/'info.json').read_text())
    g=np.load(b/'groups.npy');y=np.load(b/'y.npy')
    raw=np.load(b/'x.npy',mmap_mode='r');mask=np.load(b/'mask.npy',mmap_mode='r');public=np.load(b/'foundation.npy',mmap_mode='r')
    k=len(info['label_values'])
    cls={'public_target':TargetDescriptors,'public_latency':LatencyDescriptors}[kind]
    def fit(rows):return cls(info).to(device).fit(raw,mask,y,rows,device,g)
    def embeddings(rows):
        return torch.cat([global_summary(torch.tensor(public[ix].astype('float32'),device=device),info)
                          for ix in np.array_split(rows,(len(rows)+511)//512)])
    xs=[];ys=[];ps=[]
    for held in split_groups(y,g,tr,3):
        sub=np.setdiff1d(tr,held)
        if np.intersect1d(g[sub],g[held]).size:raise ValueError('Source overlap inside reference crossfit')
        xs.append(transform(fit(sub),held,raw,mask,device));ys.append(y[held]);ps.append(embeddings(held))
    proto=fit(tr);x=torch.cat(xs).to(device);p=torch.cat(ps);labels=torch.tensor(np.concatenate(ys),device=device)
    if len(va):
        v=transform(proto,va,raw,mask,device).to(device);q=embeddings(va)
    head=PublicResponseHead(x.shape[-1],k).to(device)
    head.center.copy_(x.mean((0,1)));head.scale.copy_(x.std((0,1)).clamp_min(.012).reciprocal())
    head.pub_center.copy_(p.mean(0));head.pub_scale.copy_(p.std(0).clamp_min(.05).reciprocal())
    opt=torch.optim.AdamW(head.parameters(),lr=.0013,weight_decay=.03)
    curve=[]
    for ep in range(epochs):
        head.train()
        for group in opt.param_groups:group['lr']=.0013/(1+.10*ep)
        for r in torch.randperm(len(labels),device=device).split(192):
            z=head(x[r],p[r]);loss=torch.nn.functional.cross_entropy(z,labels[r],label_smoothing=.045)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if len(va):
            head.eval()
            with torch.inference_mode():
                scores=torch.cat([head(v[i:i+192],q[i:i+192]) for i in range(0,len(v),192)]).cpu().numpy()
            curve.append(float(balanced_accuracy_score(y[va],scores.argmax(1))))
    out.mkdir(exist_ok=True,parents=True)
    torch.save(dict(kind=kind,dataset=task,dim=x.shape[-1],proto=proto.state_dict(),head=head.state_dict()),out/'model.pt')
    if len(va):
        with torch.inference_mode(): blank=head(v[:24],head.pub_center.expand(min(24,len(v)),-1))
        dependence=float((blank.cpu()-torch.tensor(scores[:24])).abs().max())
        if dependence<=1e-5:raise ValueError('Public encoder has no measurable contribution')
        np.savez_compressed(out/'val.npz',ids=va,logits=scores)
    else:dependence=None
    (out/'metrics.json').write_text(json.dumps(dict(epoch=epochs,curve=curve,public_ablation_difference=dependence)))
    return curve
