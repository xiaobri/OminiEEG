"""Task-specific trainable spatial and temporal adapters for a public EEG encoder."""
import math
import torch
from torch import nn
import torch.nn.functional as F

def compact_public(z,info):
    c,t=info['input_shapes'][0]
    a=min(c,10);b=min(3,max(1,round(t/info['sampling_rate_hz'])))
    v=z.reshape(len(z),2,200,a,b)
    # Public representations retain regional information in addition to their mean.
    return torch.cat([v.mean((-1,-2)).flatten(1),F.adaptive_avg_pool1d(v[:,-1].mean(-1),min(4,a)).flatten(1)],1)

class TemporalBlock(nn.Module):
    def __init__(self,w,k,dil):
        super().__init__()
        self.seq=nn.Sequential(nn.Conv1d(w,w,k,padding=dil*(k//2),dilation=dil,groups=w,bias=False),
                               nn.BatchNorm1d(w),nn.GELU(),nn.Conv1d(w,w,1,bias=False),nn.BatchNorm1d(w))
    def forward(self,x):
        return F.gelu(self.seq(x)+x)

class EEGAdapter(nn.Module):
    def __init__(self,info,dim):
        super().__init__()
        self.family=info['dataset'].split('_')[0]
        c,t=info['input_shapes'][0];fs=info['sampling_rate_hz']
        self.register_buffer('center',torch.zeros(1,c,1));self.register_buffer('inv',torch.ones(1,c,1))
        self.register_buffer('pm',torch.zeros(dim));self.register_buffer('ps',torch.ones(dim))
        self.down=max(1,round(fs/200));self.sleep=self.family=='sleep'
        t=t//self.down
        bands=[(.4,4),(4,8),(8,12),(12,18),(18,27),(27,42),(42,72)]
        self.n=min(t,400); ff=torch.fft.rfftfreq(self.n,1/(fs/self.down))
        self.ix=[(int(torch.searchsorted(ff,lo)),int(torch.searchsorted(ff,hi))) for lo,hi in bands]
        self.register_buffer('taper',torch.hann_window(self.n))
        self.fb=nn.Parameter(torch.randn(len(bands),12,c)/math.sqrt(c))
        self.t_bins=min(4,max(1,(t-self.n)//(self.n//2)+1))
        self.band_head=nn.Sequential(nn.Flatten(),nn.Linear(len(bands)*12*self.t_bins,96),nn.LayerNorm(96),nn.GELU())
        w=48
        self.spatial=nn.Conv1d(c,w,1,bias=False)
        self.temporal=nn.Sequential(nn.Conv1d(w,w,17,stride=4 if self.sleep else 2,padding=8,groups=w,bias=False),
                                    nn.BatchNorm1d(w),nn.GELU(),TemporalBlock(w,15,1),nn.AvgPool1d(3,2,padding=1),
                                    TemporalBlock(w,11,2),TemporalBlock(w,7,4))
        self.pub=nn.Sequential(nn.Dropout(.25),nn.Linear(dim,96),nn.LayerNorm(96),nn.GELU())
        bins=8 if self.family in ('sleep','stimulus','ssvep') else 4
        self.bins=bins
        self.classify=nn.Sequential(nn.Dropout(.32),nn.Linear(w*(bins+1)+192,128),nn.LayerNorm(128),nn.GELU(),
                                    nn.Dropout(.3),nn.Linear(128,len(info['label_values'])))
    def forward(self,x,mask,public):
        x=((torch.nan_to_num(x.float(),nan=0.,posinf=0.,neginf=0.)-self.center)*self.inv).clamp(-15,15)*mask[:,:,None]
        if self.training:
            x=x*torch.empty((len(x),1,1),device=x.device).normal_(0,.16).exp()
            if x.shape[1]>2:
                keep=torch.rand((len(x),x.shape[1],1),device=x.device)>.06
                x=x*keep/.94
        if self.down>1:x=F.avg_pool1d(x,self.down,self.down)
        ft=torch.fft.rfft(x.unfold(-1,self.n,self.n//2)*self.taper)/math.sqrt(self.n)
        e=[]
        for j,(lo,hi) in enumerate(self.ix):
            # Project both quadratures with the same learned spatial filters.
            f=ft[...,lo:hi]
            w=F.normalize(self.fb[j],dim=-1)
            re=torch.einsum('lc,bcsf->blsf',w,f.real)
            im=torch.einsum('lc,bcsf->blsf',w,f.imag)
            power=(re.square()+im.square()).mean(-1).clamp_min(1e-7).log()
            e.append(F.adaptive_avg_pool1d(power,self.t_bins))
        freq=self.band_head(torch.cat(e,1))
        wave=self.temporal(self.spatial(x))
        pooled=F.adaptive_avg_pool1d(wave,self.bins).flatten(1)
        pub=self.pub(((public-self.pm)*self.ps).clamp(-8,8))
        return self.classify(torch.cat([pooled,wave.amax(-1),freq,pub],dim=1))
