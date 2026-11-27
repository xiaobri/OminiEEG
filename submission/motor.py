"""Source-trained filter-bank spatial projections for motor imagery."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Motor(nn.Module):
    def __init__(self,info):
        super().__init__()
        self.classes=len(info['label_values'])
        c,t=info['input_shapes'][0]
        names=[str(n).strip('.').upper() for n in info['channel_names']]
        sensorimotor=[i for i,name in enumerate(names) if name.startswith(('FC','C','CP')) and not name.startswith('CB')]
        if len(sensorimotor)<6:
            sensorimotor=list(range(c))
        self.register_buffer('channels',torch.tensor(sensorimotor))
        freq=torch.fft.rfftfreq(t,1/info['sampling_rate_hz'])
        bands=[(5,9),(8,12),(10,14),(13,19),(17,23),(21,29),(27,36),(7,32)]
        self.register_buffer('filters',torch.stack([torch.sigmoid((freq-lo)*2)*torch.sigmoid((hi-freq)*2) for lo,hi in bands]))
        self.register_buffer('spatial',torch.zeros(len(bands),len(sensorimotor),self.classes*4))
        self.t=t

    def waves(self,x,mask):
        # Common average reference and specified scalp montage for each individual window.
        x=x.float()*mask[:,:,None]
        x=(x[:,self.channels]-x.sum(1,keepdim=True)/mask.sum(1).clamp_min(1)[:,None,None])*mask[:,self.channels,None]
        x=x-x.mean(-1,keepdim=True)
        return torch.fft.irfft(torch.fft.rfft(x)[:,None]*self.filters[:,None],n=self.t)

    @torch.no_grad()
    def fit(self,x,m,y,g,ids,device,seed):
        rng=np.random.default_rng(seed)
        bands,c,_=self.spatial.shape
        stat=torch.zeros(self.classes,bands,c,c,dtype=torch.float64,device=device)
        for label in range(self.classes):
            rows=ids[y[ids]==label]
            subjects=np.unique(g[rows])
            for sub in subjects:
                ix=rows[g[rows]==sub]
                ix=rng.choice(ix,min(len(ix),180),replace=False)
                for batch in np.array_split(ix,(len(ix)+47)//48):
                    h=self.waves(torch.as_tensor(x[batch],device=device),torch.as_tensor(m[batch],device=device))
                    h=h[...,self.t//5:]
                    cov=h@h.mT
                    cov/=cov.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-8)[...,None,None]
                    stat[label]+=cov.sum(0,dtype=torch.float64)/(len(ix)*len(subjects))
        total=stat.mean(0)
        eig,vec=torch.linalg.eigh(.86*total+.14*torch.eye(c,device=device)/c)
        white=(vec*eig.clamp_min(1e-9).rsqrt().unsqueeze(-2))@vec.mT
        _,vec=torch.linalg.eigh(white@stat@white)
        basis=white@torch.cat((vec[...,:2],vec[...,-2:]),dim=-1)
        self.spatial.copy_(basis.permute(1,2,0,3).flatten(2))
        return self

    def forward(self,x,mask):
        waves=self.waves(x,mask)
        projected=torch.einsum('bfct,fcr->bfrt',waves,self.spatial)
        power=F.adaptive_avg_pool1d(projected.square().flatten(1,2),4).clamp_min(1e-8).log()
        power=power.reshape(len(x),len(self.filters),-1,4)
        local=F.adaptive_avg_pool1d(waves.square().flatten(1,2),4).clamp_min(1e-8).log()
        local=local.reshape(len(x),len(self.filters),-1,4)
        return torch.cat([v.flatten(1) for v in (power.mean(-1),power-power.mean(-2,keepdim=True),
                             power-power.mean(-1,keepdim=True),local-local.mean(-2,keepdim=True))],1)
