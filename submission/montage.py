"""Learn a task montage from all release channels before public CBraMod fine-tuning."""
import torch
from torch import nn
from torch.nn import functional as F

from foundation import Foundation


class Montage(nn.Module):
    def __init__(self,info,weights):
        super().__init__()
        self.backbone=Foundation(info,weights)
        channels,length=info['input_shapes'][0]
        self.fs=info['sampling_rate_hz']
        self.size=min(24,channels)
        initial=torch.zeros(self.size,channels)
        # Keep coverage across the published channel list at initialization.
        initial.scatter_(1,torch.linspace(0,channels-1,self.size).round().long()[:,None],1.)
        self.register_buffer('initial',initial)
        self.offset=nn.Parameter(torch.zeros_like(initial))
        self.gain=nn.Parameter(torch.zeros(self.size))
        self.backbone.requires_grad_(False)
        self.backbone.encoder.encoder.layers[-4:].requires_grad_(True)
        self.backbone.encoder.patch_embedding.requires_grad_(True)
        self.backbone.encoder.patch_embedding.mask_encoding.requires_grad_(False)
        self.read=nn.Sequential(nn.LayerNorm(200),nn.Linear(200,40),nn.GELU())
        self.pc=min(8,self.size)
        self.pt=min(5,max(1,round(2*length/self.fs)-1))
        self.head=nn.Sequential(nn.Flatten(),nn.LayerNorm(self.pc*self.pt*40),nn.Dropout(.30),
                                nn.Linear(self.pc*self.pt*40,144),nn.LayerNorm(144),nn.GELU(),
                                nn.Dropout(.32),nn.Linear(144,len(info['label_values'])))

    def forward(self,x,mask):
        x=x.float()*mask[:,:,None]
        if self.training:
            if x.shape[1]>3:
                x=x*(torch.rand(*x.shape[:2],1,device=x.device)>.06)/.94
            x=x*torch.randn(len(x),1,1,device=x.device).mul(.18).exp()
        montage=F.normalize(self.initial+self.offset,dim=1)*self.gain.clamp(-2.,2.).exp()[:,None]
        x=(montage@x).float()
        if self.fs!=200:
            x=F.adaptive_avg_pool1d(x,max(200,round(x.shape[-1]*200/self.fs)))
        if x.shape[-1]<200:
            x=F.pad(x,(0,200-x.shape[-1]))
        patches=(x*.01).clamp(-30,30).unfold(-1,200,100).contiguous()
        h=self.backbone.encoder.patch_embedding(patches)
        for layer in self.backbone.encoder.encoder.layers:
            h=layer(h)
        v=self.read(h).permute(0,3,1,2)
        return self.head(F.adaptive_avg_pool2d(v,(self.pc,self.pt)))

    def delta(self):
        return {name:p.detach().cpu() for name,p in self.named_parameters() if p.requires_grad}
