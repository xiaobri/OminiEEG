"""Single-window EEG feature maps and supervised networks for the public tasks."""
import math
import torch
from torch import nn
import torch.nn.functional as F

BANDS=((0.5,2),(2,4),(4,8),(8,12),(12,16),(16,23),(23,30),(30,45),(45,57),(57,85),(85,150),(150,350))

class Features(nn.Module):
    def __init__(self, info):
        super().__init__()
        self.task=info['dataset']; self.family=self.task.rsplit('_',1)[0]
        self.c,self.t=info['input_shapes'][0]
        fs=float(info['sampling_rate_hz']); self.fs=fs
        n=min(self.t,int(2*fs));self.n=n
        self.register_buffer('window',torch.hann_window(n))
        self.bands=[]
        for a,b in BANDS:
            lo=max(1,math.ceil(a*n/fs));hi=min(n//2+1,math.ceil(b*n/fs))
            if hi>lo:self.bands.append((lo,hi))
        sel=torch.linspace(0,self.c-1,min(self.c,24)).round().long()
        if self.family=='ssvep':
            occ=[i for i,k in enumerate(info['channel_names']) if str(k).upper().startswith(('PO','O','PZ','P3','P4'))]
            if occ:sel=torch.tensor(occ)
        self.register_buffer('sel',sel)
        self.register_buffer('tri',torch.triu_indices(len(sel),len(sel),offset=1))
        if self.family=='ssvep':
            times=torch.arange(self.t)/fs
            freq=torch.arange(7.0,16.81,0.2)
            phase=(2*math.pi*freq[:,None,None]*torch.arange(1,5)[None,:,None]*times).flatten(0,1)
            self.register_buffer('ref',torch.cat((phase.cos(),phase.sin()))/math.sqrt(self.t))
    def forward(self, x, mask):
        x=torch.nan_to_num(x.float(),nan=0.,posinf=0.,neginf=0.).clamp(-1e8,1e8)*mask.unsqueeze(-1)
        x=x-x.mean(-1,keepdim=True)
        chunks=x.unfold(-1,self.n,max(1,self.n//2))
        f=torch.fft.rfft(chunks*self.window, dim=-1)/math.sqrt(self.n)
        p=f.abs().square()
        rms=x.square().mean(-1).clamp_min(1e-12).sqrt()
        z=x/rms.unsqueeze(-1)
        pieces=[rms.log(),z.abs().mean(-1),z.abs().amax(-1).clamp_max(100),
                z.clamp(-12,12).pow(3).mean(-1),z.clamp(-12,12).pow(4).mean(-1).log1p(),
                z.diff(dim=-1).abs().mean(-1)]
        powers=torch.stack([p[...,a:b].mean(-1).clamp_min(1e-12).log() for a,b in self.bands],dim=-1)
        pieces.extend([powers.mean(-2),powers.amax(-2),powers.std(-2,unbiased=False)])
        if self.family in ('sleep','reliability'):
            avg=p.mean(-2)
            pieces.append(F.adaptive_avg_pool1d(avg,64).clamp_min(1e-12).log())
            pieces.append(F.adaptive_avg_pool1d(x.square(),12).clamp_min(1e-12).log())
        if self.c>1:
            f=f[:,self.sel]
            for a,b in self.bands:
                v=f[...,a:b].flatten(2)
                denom=v.abs().square().sum(-1).clamp_min(1e-12).rsqrt()
                v=v*denom.unsqueeze(-1)
                cov=(v@v.mH).real
                pieces.append(cov[:,self.tri[0],self.tri[1]])
        if self.family in ('stimulus','motor','ssvep'):
            pieces.append(F.adaptive_avg_pool1d(z[:,self.sel],32))
        if self.family=='ssvep':
            harmonic=z[:,self.sel]@self.ref.T
            real,imag=harmonic.chunk(2,dim=-1)
            pieces.extend((real,imag,torch.log1p(real.square()+imag.square())))
        return torch.cat([v.reshape(x.shape[0],-1) for v in pieces],dim=1)

class Head(nn.Module):
    def __init__(self,d,k,kind,hidden=144):
        super().__init__()
        self.register_buffer('mean',torch.zeros(d))
        self.register_buffer('scale',torch.ones(d))
        if kind=='linear': self.layers=nn.Linear(d,k)
        else:
            self.layers=nn.Sequential(nn.Dropout(0.18),nn.Linear(d,hidden),nn.LayerNorm(hidden),
                                      nn.GELU(),nn.Dropout(0.35),nn.Linear(hidden,k))
    def forward(self,v):
        return self.layers(((v-self.mean)*self.scale).clamp(-8,8))
