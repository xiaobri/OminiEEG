"""One response classifier conditioned on a frozen public EEG encoder embedding."""
import torch
from torch import nn


class PublicResponseHead(nn.Module):
    def __init__(self,dim,classes):
        super().__init__()
        self.register_buffer('center',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.register_buffer('pub_center',torch.zeros(400));self.register_buffer('pub_scale',torch.ones(400))
        self.embed=nn.Sequential(nn.Linear(400,64),nn.LayerNorm(64),nn.GELU(),nn.Dropout(.2))
        self.gate=nn.Linear(64,dim)
        self.class_context=nn.Linear(64,classes*6)
        self.scorer=nn.Sequential(nn.Dropout(.12),nn.Linear(2*dim+6,64),nn.LayerNorm(64),nn.GELU(),
                                  nn.Dropout(.15),nn.Linear(64,1))
        self.classes=classes

    def forward(self,local,public):
        p=self.embed(((public-self.pub_center)*self.pub_scale).clamp(-7,7))
        x=((local-self.center)*self.scale).clamp(-8,8)
        z=x*(1+.25*self.gate(p).tanh()[:,None])
        relative=(z-z.mean(1,keepdim=True))/z.std(1,unbiased=False,keepdim=True).clamp_min(.30)
        return self.scorer(torch.cat((z,relative.clamp(-8,8),self.class_context(p).reshape(len(p),self.classes,6)),dim=-1)).squeeze(-1)
