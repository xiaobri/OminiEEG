"""Estimate a common training latency for each source subject and one target template bank."""
import numpy as np
import torch
from torch.nn import functional as F
from response_decoder import ResponseDescriptors


class LatencyDescriptors(ResponseDescriptors):
    def __init__(self,info):
        super().__init__(info)
        n=len(info['label_values']);length=info['input_shapes'][0][1]
        self.padding=max(1,round(.06*info['sampling_rate_hz']))
        self.register_buffer('reference',torch.zeros(n,5,6,length))

    @torch.no_grad()
    def fit(self,x,m,y,ids,device,groups):
        super().fit(x,m,y,ids,device,groups)
        means=[]
        for subject in np.unique(groups[ids]):
            rows=ids[groups[ids]==subject]
            total=torch.zeros_like(self.reference);count=torch.zeros(len(total),device=device)
            for i in range(0,len(rows),64):
                r=rows[i:i+64];target=torch.from_numpy(y[r]).to(device)
                wave=self.population.transform.mT@self.population.bank(torch.from_numpy(x[r]).to(device),torch.from_numpy(m[r]).to(device))
                total.index_add_(0,target,wave);count+=torch.bincount(target,minlength=len(total))
            total/=count.clamp_min(1)[:,None,None,None]
            total-=total.mean(0,keepdim=True)
            means.append(total / total.square().mean((0,2,3),keepdim=True).clamp_min(1e-7).sqrt())
        waves=torch.stack(means)
        ref=waves.mean(0)
        shifts=range(-self.padding,self.padding+1,2)
        # Fit one scalar latency per training subject across all target classes.
        for _ in range(2):
            later=[]
            for subject in waves:
                v=F.normalize(ref[:,:3,:,self.padding:-self.padding].flatten(2),dim=-1)
                merit=[]
                for lag in shifts:
                    cropped=subject.roll(lag,-1)[:,:3,:,self.padding:-self.padding]
                    merit.append((F.normalize(cropped.flatten(2),dim=-1)*v).sum(-1).mean()-.0003*abs(lag))
                lag=shifts[int(torch.stack(merit).argmax())]
                later.append(subject.roll(lag,-1))
            ref=torch.stack(later).mean(0)
        self.reference.copy_(ref)
        return self

    def forward(self,x,mask):
        pieces=[super().forward(x,mask)]
        wave=self.population.transform.mT@self.population.bank(x,mask)
        for begin in (self.padding,wave.size(-1)//3):
            for rank in (2,6):
                template=F.normalize(self.reference[:,:,-rank:,begin:-self.padding].flatten(-2),dim=-1)
                scores=[]
                for lag in (-self.padding,-self.padding//2,0,self.padding//2,self.padding):
                    h=F.normalize(wave.roll(lag,-1)[:,:,-rank:,begin:-self.padding].flatten(-2),dim=-1)
                    scores.append(torch.einsum('bfd,kfd->bkf',h,template))
                s=torch.stack(scores,-1)
                pieces.extend((s[...,2],s.amax(-1),s.mean(-1),s[...,1]-s[...,3]))
        return torch.cat(pieces,-1)
