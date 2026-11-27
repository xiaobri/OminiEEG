"""Training-only spatial filters and class templates for steady-state responses."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Steady(nn.Module):
    def __init__(self, info):
        super().__init__()
        names=info['channel_names'];c,t=info['input_shapes'][0];fs=info['sampling_rate_hz']
        channels=[i for i,s in enumerate(names) if s.upper().startswith(('PO','O','PZ','P1','P2','P3','P4'))]
        if len(channels)<4:channels=list(range(max(0,c-14),c))
        self.register_buffer('sel',torch.tensor(channels))
        k=len(info['label_values']);c=len(channels);self.k=k;self.t=t;self.r=min(8,c)
        freq=torch.fft.rfftfreq(t,1/fs)
        responses=torch.stack([torch.sigmoid((freq-cut)*1.8)*torch.sigmoid((88-freq)*1.8) for cut in [6.,14.,22.,30.,38.]])
        self.register_buffer('filters',responses)
        # A densely sampled reference grid. Mapping to classes is learned from training signals.
        fftlen=2000
        idx=(torch.arange(8,16,0.2,dtype=torch.float64)[:,None]*torch.arange(1,6)*fftlen/fs).round().long()
        self.register_buffer('bins',idx)
        self.register_buffer('basis',torch.zeros(5,c,self.r))
        self.register_buffer('templates',torch.zeros(5,k,self.r,t))
        self.register_buffer('order',torch.arange(k))
        self.register_buffer('power_ref',torch.ones(40,5))
        self.register_buffer('window',torch.hann_window(t))
    def raw(self,x,mask):
        x=x[:,self.sel].float()*mask[:,self.sel,None]
        x=x-x.mean(-1,keepdim=True)
        return x/x.square().mean((1,2),keepdim=True).clamp_min(1e-10).sqrt()
    def bank(self,x):
        return torch.fft.irfft(torch.fft.rfft(x)[:,None]*self.filters[None,:,None],n=self.t)
    def harmonics(self,x):
        spec=torch.fft.rfft(x*self.window,n=2000)
        return (spec.abs().square()[:,:,self.bins]/self.t).mean(1)
    @torch.no_grad()
    def fit(self,xb,mb,y,ids,device):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        c=len(self.sel);n=len(ids)
        sums=torch.zeros(5,self.k,c,self.t,dtype=torch.float64,device=device)
        cov=torch.zeros(5,c,c,dtype=torch.float64,device=device)
        power=torch.zeros(self.k,40,5,dtype=torch.float64,device=device)
        count=torch.zeros(self.k,dtype=torch.float64,device=device)
        for at in range(0,n,96):
            ix=ids[at:at+96]
            raw=self.raw(torch.from_numpy(xb[ix]).to(device),torch.from_numpy(mb[ix]).to(device))
            label=torch.as_tensor(y[ix],device=device)
            b=self.bank(raw).double()
            sums.index_add_(1,label,b.transpose(0,1));cov+=torch.einsum('bfct,bfdt->fcd',b,b)
            power.index_add_(0,label,self.harmonics(raw).clamp_min(1e-8).log().double())
            count+=torch.bincount(label,minlength=self.k)
        assert count.min()>0
        means=sums/count[None,:,None,None]
        cov/=n*self.t
        within_mean=torch.einsum('fkct,fkdt->fcd',means,means)/(self.k*self.t)
        eig,vec=torch.linalg.eigh(cov+0.07*cov.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*torch.eye(c,device=device))
        w=(vec*eig.clamp_min(1e-8).rsqrt()[:,None,:])@vec.mT
        _,u=torch.linalg.eigh(w@within_mean@w)
        basis=w@u[:,:,-self.r:]
        self.basis.copy_(basis)
        self.templates.copy_(torch.einsum('fcr,fkct->fkrt',basis,means))
        power/=count[:,None,None];background=power.mean(0)
        weight=torch.arange(1,6,device=device).rsqrt()
        scores=((power-background)*weight).sum(-1)
        _,order=linear_sum_assignment(-scores.cpu().numpy())
        self.order.copy_(torch.as_tensor(order));self.power_ref.copy_(background)
        return self
    def forward(self,x,mask):
        raw=self.raw(x,mask);b=self.bank(raw)
        proj=torch.einsum('bfct,fcr->bfrt',b,self.basis)
        t=self.templates
        # Fixed lags allow a few samples of onset jitter without fitting on a participant.
        vals=[]
        for offset in (-6,-3,0,3,6):
            tv=t.roll(offset,dims=-1)
            norm=F.normalize(tv.flatten(-2),dim=-1,eps=1e-7)
            corr=torch.einsum('bfn,fkn->bkf',F.normalize(proj.flatten(-2),dim=-1,eps=1e-7),norm)
            vals.append(corr)
        corr=torch.stack(vals,-1)
        spectral=self.harmonics(raw).clamp_min(1e-8).log()-self.power_ref
        return torch.cat([corr.flatten(2),corr.amax(-1),spectral[:,self.order]],dim=-1)

class SteadyHead(nn.Module):
    def __init__(self,local,globaldim,classes):
        super().__init__()
        self.register_buffer('m',torch.zeros(local));self.register_buffer('s',torch.ones(local))
        self.register_buffer('gm',torch.zeros(globaldim));self.register_buffer('gs',torch.ones(globaldim))
        self.local=nn.Sequential(nn.Linear(local,32),nn.GELU(),nn.Linear(32,1))
        self.global_linear=nn.Linear(globaldim,classes)
    def forward(self,x,g):
        return self.local(((x-self.m)*self.s).clamp(-8,8)).squeeze(-1)+self.global_linear(((g-self.gm)*self.gs).clamp(-8,8))

def global_summary(z,info):
    c,t=info['input_shapes'][0];a=min(c,10);b=min(3,max(1,round(t/info['sampling_rate_hz'])))
    return z.reshape(len(z),2,200,a,b).mean((-1,-2)).flatten(1)
