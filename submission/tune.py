"""Fine-tune the final CBraMod blocks with spatially resolved downstream readout."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from foundation import Foundation

class Tuned(nn.Module):
    def __init__(self,info,weights):
        super().__init__()
        self.base=Foundation(info,weights)
        self.base.requires_grad_(False)
        self.base.encoder.encoder.layers[-2:].requires_grad_(True)
        c=len(self.base.channels);pc=min(12,c)
        t=info['input_shapes'][0][1]; pt=min(6,max(1,round(t/info['sampling_rate_hz'])))
        self.pools=(pc,pt)
        self.local=nn.Sequential(nn.LayerNorm(200),nn.Linear(200,40),nn.GELU())
        self.readout=nn.Sequential(nn.Flatten(),nn.LayerNorm(pc*pt*40),nn.Dropout(.35),nn.Linear(pc*pt*40,160),
                                   nn.LayerNorm(160),nn.GELU(),nn.Dropout(.35),nn.Linear(160,len(info['label_values'])))
    def frozen(self,x,mask):
        p=self.base.encoder.patch_embedding(self.base.patches(x,mask))
        for layer in self.base.encoder.encoder.layers[:-2]:p=layer(p)
        return p.half()
    def classify(self,h):
        h=h.float()
        if self.training and h.shape[1]>3:
            keep=torch.rand(h.shape[0],h.shape[1],1,1,device=h.device)>.06
            h=h*keep/.94
        for block in self.base.encoder.encoder.layers[-2:]:h=block(h)
        return self.readout(F.adaptive_avg_pool2d(self.local(h).permute(0,3,1,2),self.pools))
    def forward(self,x,mask):
        return self.classify(self.frozen(x,mask))
    def delta(self):
        return {k:v.detach().cpu() for k,v in self.named_parameters() if v.requires_grad}

