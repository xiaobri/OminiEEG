"""Filter-bank covariance features with source-trained whitening and spatial filters."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class Geometry(nn.Module):
    def __init__(self,info):
        super().__init__()
        c,t=info['input_shapes'][0];fs=info['sampling_rate_hz'];self.t=t
        self.k=len(info['label_values']);self.r=min(c,20)
        bands=[(.5,4),(4,8),(8,13),(13,20),(20,30),(30,48),(.5,48)]
        self.n=min(t,round(2*fs));self.freqbands=[(max(1,round(a*self.n/fs)),max(2,round(b*self.n/fs))) for a,b in bands]
        self.register_buffer('win',torch.hann_window(self.n))
        self.register_buffer('basis',torch.zeros(len(bands),c,self.r))
        self.register_buffer('tri',torch.triu_indices(self.r,self.r))
    def cov(self,x,m):
        x=x.float()*m[:,:,None]
        x=x-x.mean(-1,keepdim=True)
        z=torch.fft.rfft(x.unfold(-1,self.n,self.n//2)*self.win)/self.n
        cov=[]
        for lo,hi in self.freqbands:
            h=z[...,lo:hi].flatten(2)
            v=(h@h.mH).real/max(1,h.shape[-1])
            trace=torch.diagonal(v,dim1=-2,dim2=-1).mean(-1).clamp_min(1e-12)
            cov.append(v/trace[:,None,None])
        return torch.stack(cov,1)
    @torch.no_grad()
    def fit(self,x,m,y,g,tr,device,seed):
        rng=np.random.default_rng(seed);f,c,r=self.basis.shape
        stats=torch.zeros(self.k,f,c,c,dtype=torch.float64,device=device)
        for label in range(self.k):
            ids=tr[y[tr]==label];subjects=np.unique(g[ids]);select=[];weights=[]
            for s in subjects:
                ix=ids[g[ids]==s];cnt=min(len(ix),max(12,min(128,2400//len(subjects))))
                select.extend(rng.choice(ix,cnt,replace=False));weights.extend([1/(cnt*len(subjects))]*cnt)
            for j in range(0,len(select),64):
                sub=select[j:j+64];v=self.cov(torch.as_tensor(x[sub],device=device),torch.as_tensor(m[sub],device=device))
                stats[label]+=torch.einsum('b,bfcd->fcd',torch.tensor(weights[j:j+64],dtype=torch.float64,device=device),v.double())
        mean=stats.mean(0);eye=torch.eye(c,device=device)
        ev,u=torch.linalg.eigh(.90*mean+.10*eye)
        white=(u*ev.clamp_min(1e-6).rsqrt()[:,None,:])@u.mT
        within=white[None]@stats@white[None]
        d=within-within.mean(0,keepdim=True)
        contrast=(d@d).mean(0)
        _,eig=torch.linalg.eigh(contrast)
        self.basis.copy_(white@eig[:,:,-r:])
        return self
    def forward(self,x,m):
        all_cov=self.cov(x,m)
        cov=self.basis.mT[None]@all_cov@self.basis[None]
        # Shrink each within-window covariance before mapping it into tangent space.
        eye=torch.eye(self.r,device=x.device)
        ev,u=torch.linalg.eigh(.94*cov+.06*eye)
        log=(u*ev.clamp_min(.001).log().unsqueeze(-2))@u.mT
        return log[:,:,self.tri[0],self.tri[1]].flatten(1)
