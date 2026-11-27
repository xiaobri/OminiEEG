"""Latency-aware templates and source-only harmonic spatial projection."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from steady import Steady

class SteadyWindow(Steady):
    def __init__(self,info):
        super().__init__(info);c=len(self.sel);h=5
        self.register_buffer('weights',torch.ones(5,self.t))
        self.register_buffer('spatial',torch.zeros(40,h,c))
        self.register_buffer('spec_base',torch.zeros(40,h))
        self.register_buffer('freq_order',torch.arange(40))
    def assign_grid(self,merit):
        from scipy.optimize import linear_sum_assignment
        n=np.arange(40);layouts=[]
        for st in [n,n.reshape(5,8).T.flatten(),n.reshape(8,5).T.flatten()]:
            for sign in [1,-1]:
                for lag in range(40):layouts.append((sign*st+lag)%40)
        prior=max(layouts,key=lambda a:merit[n,a].sum());bonus=np.zeros_like(merit);bonus[n,prior]=max(.12,merit.std()*.75)
        return linear_sum_assignment(-(merit+bonus))[1]
    @torch.no_grad()
    def fit(self,xb,mb,y,ids,device):
        super().fit(xb,mb,y,ids,device)
        t=self.templates
        # Remove the time-locked waveform common to all target classes.
        t-=t.mean(1,keepdim=True)
        power=t.square().mean((1,2))
        smoothed=F.avg_pool1d(F.pad(power[:,None],(10,10),mode='replicate'),21,stride=1).squeeze(1)
        self.weights.copy_((smoothed/smoothed.mean(-1,keepdim=True).clamp_min(1e-10)).clamp(.12,4).sqrt())
        c=len(self.sel);counts=torch.zeros(40,device=device,dtype=torch.float64)
        stat=torch.zeros(40,40,5,c,c,dtype=torch.float64,device=device)
        for at in range(0,len(ids),64):
            ix=ids[at:at+64];raw=self.raw(torch.as_tensor(xb[ix],device=device),torch.as_tensor(mb[ix],device=device))
            z=torch.fft.rfft(raw*self.weights[0]*self.window,n=2000)[:,:,self.bins].permute(0,2,3,1)
            z=z/(raw.shape[-1]**.5);cov=z.real[...,None]*z.real[...,None,:]+z.imag[...,None]*z.imag[...,None,:]
            label=torch.as_tensor(y[ix],device=device)
            stat.index_add_(0,label,cov.double());counts.index_add_(0,label,torch.ones_like(label,dtype=torch.float64))
        stat/=counts[:,None,None,None,None];ref=stat.mean(0)
        eye=torch.eye(c,device=device)
        ref=.85*ref+.15*ref.diagonal(dim1=-2,dim2=-1).mean(-1)[:,:,None,None]*eye
        v,u=torch.linalg.eigh(ref);white=(u*v.clamp_min(1e-8).rsqrt().unsqueeze(-2))@u.mT
        cov=white[None]@stat@white[None];v,u=torch.linalg.eigh(cov)
        merit=v[...,-1].clamp_min(.001).log().sum(-1).cpu().numpy()
        # Regularize the assignment toward a stimulation grid when supported by source data.
        order=torch.as_tensor(self.assign_grid(merit),device=device)
        self.freq_order.copy_(order)
        ix=torch.arange(40,device=device);top=u[ix,order,:,:,-1]
        f=(white[order]@top.unsqueeze(-1)).squeeze(-1);self.spatial.copy_(f)
        ps=torch.einsum('khc,khcd,khd->kh',f,ref[order],f)
        self.spec_base.copy_(ps.clamp_min(1e-8).log())
        return self
    def forward(self,x,mask):
        raw=self.raw(x,mask);p=torch.einsum('bfct,fcr->bfrt',self.bank(raw),self.basis);t=self.templates
        pieces=[]
        for shrink in (0.,.35,.58):
            start=round(self.t*shrink)
            w=self.weights[None,:,None,start:]
            h=F.normalize((p[...,start:]*w).flatten(-2),dim=-1,eps=1e-7)
            block=[]
            for lag in (-5,-2,0,2,5):
                template=t.roll(lag,dims=-1)[...,start:]*w.transpose(0,1)
                corr=torch.einsum('bfn,fkn->bkf',h,F.normalize(template.flatten(-2),dim=-1,eps=1e-7))
                block.append(corr)
            v=torch.stack(block,-1);pieces.extend((v.flatten(2),v.amax(-1)))
        for win in (self.window*self.weights[0],self.weights[0]):
            z=torch.fft.rfft(raw*win,n=2000)[:,:,self.bins[self.freq_order]].permute(0,2,3,1)
            re=(z.real*self.spatial).sum(-1);im=(z.imag*self.spatial).sum(-1)
            pieces.append(((re.square()+im.square())/self.t).clamp_min(1e-8).log()-self.spec_base)
        return torch.cat(pieces,dim=-1)
