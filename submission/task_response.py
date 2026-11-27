"""Target-specific correlated spatial directions estimated across independent source subjects."""
import numpy as np
import torch
from torch.nn import functional as F
from latency_response import LatencyDescriptors


class TargetDescriptors(LatencyDescriptors):
    def __init__(self,info):
        super().__init__(info)
        k=len(info['label_values']);c=len(self.population.channels);n=self.population.length
        self.register_buffer('spatial_by_target',torch.zeros(k,5,c,3))
        self.register_buffer('template_by_target',torch.zeros(k,5,3,n))

    @torch.no_grad()
    def fit(self,x,m,y,ids,device,groups):
        super().fit(x,m,y,ids,device,groups)
        k,f,c,r=self.spatial_by_target.shape;n=self.population.length
        dtype=torch.float64
        cross=torch.zeros(k,f,c,c,device=device,dtype=dtype)
        within=torch.zeros_like(cross)
        total=torch.zeros(k,f,c,n,device=device,dtype=dtype)
        people=np.unique(groups[ids])
        for sub in people:
            rows=ids[groups[ids]==sub]
            sums=torch.zeros_like(total)
            gram=torch.zeros_like(cross); count=torch.zeros(k,device=device,dtype=dtype)
            for i in range(0,len(rows),48):
                ix=rows[i:i+48]; labels=torch.as_tensor(y[ix],device=device)
                signals=self.population.bank(torch.as_tensor(x[ix],device=device),torch.as_tensor(m[ix],device=device)).double()
                sums.index_add_(0,labels,signals)
                gram.index_add_(0,labels,signals@signals.mT/n)
                count+=torch.bincount(labels,minlength=k)
            if (count==0).any():raise ValueError('A source subject lacks one of the stimulation classes')
            means=sums/count[:,None,None,None]
            means-=means.mean(0,keepdim=True)
            total+=means
            cross+=means@means.mT/n
            within+=gram/count[:,None,None,None]
        a=len(people);mean=total/a
        shared=(a*(mean@mean.mT)/n-cross/a)/(a-1)
        within=within/a
        cov=.58*within+.32*within.mean(0)+.10*torch.diag_embed(within.diagonal(dim1=-1,dim2=-2))
        eig,vec=torch.linalg.eigh(cov);white=(vec*eig.clamp_min(1e-7).rsqrt().unsqueeze(-2))@vec.mT
        _,vec=torch.linalg.eigh(white@(.7*shared+.3*shared.mean(0))@white)
        basis=white@vec[...,-r:]
        self.spatial_by_target.copy_(basis.float());self.template_by_target.copy_((basis.mT@mean).float())
        return self

    def forward(self,x,mask):
        parts=[super().forward(x,mask)]
        h=self.population.bank(x,mask)
        # Compare each source-derived target subspace to this single input window.
        wave=torch.einsum('bfct,kfcr->bkfrt',h,self.spatial_by_target)
        for begin in (self.padding,wave.size(-1)//3):
            template=F.normalize(self.template_by_target[:,:,:,begin:-self.padding],dim=-1)
            lags=[]
            for lag in (-self.padding//2,0,self.padding//2):
                trial=F.normalize(wave.roll(lag,-1)[...,begin:-self.padding],dim=-1)
                lags.append((trial*template).sum(-1))
            agreement=torch.stack(lags,-1)
            parts.extend((agreement[...,1].flatten(2),agreement.amax(-1).flatten(2)))
        return torch.cat(parts,-1)
