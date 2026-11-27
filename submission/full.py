"""End-to-end fine-tuning with trainable pretrained tokens and a local readout."""
import torch
from torch import nn
from torch.nn import functional as F
from foundation import Foundation

class Full(nn.Module):
    def __init__(self,info,weights):
        super().__init__()
        self.backbone=Foundation(info,weights)
        c,t=info['input_shapes'][0]
        self.pc=min(14,len(self.backbone.channels));self.pt=min(5,max(1,round(t/info['sampling_rate_hz'])))
        self.read=nn.Sequential(nn.LayerNorm(200),nn.Linear(200,28),nn.GELU())
        self.head=nn.Sequential(nn.Flatten(),nn.LayerNorm(28*self.pc*self.pt),
                                nn.Dropout(.34),nn.Linear(28*self.pc*self.pt,144),nn.LayerNorm(144),
                                nn.GELU(),nn.Dropout(.32),nn.Linear(144,len(info['label_values'])))
        # Rank-restricted residual updates provide a compact checkpoint for every task.
        self.adapters=nn.ModuleList([nn.Sequential(nn.LayerNorm(200),nn.Linear(200,32),nn.GELU(),
                                     nn.Linear(32,200)) for _ in range(12)])
        for mod in self.adapters:
            nn.init.zeros_(mod[-1].weight);nn.init.zeros_(mod[-1].bias)
        self.backbone.requires_grad_(False)
        self.backbone.encoder.patch_embedding.requires_grad_(True)
        self.backbone.encoder.patch_embedding.mask_encoding.requires_grad_(False)
        self.task=info['dataset']
    def forward(self,x,m):
        patches=self.backbone.patches(x,m)
        if self.training:
            patches=patches*torch.empty(len(x),1,1,1,device=x.device).normal_(0,.18).exp()
            if patches.shape[1]>3:
                keep=torch.rand(len(x),patches.shape[1],1,1,device=x.device)>.06
                patches=patches*keep/.94
        h=self.backbone.encoder.patch_embedding(patches)
        for block,adapter in zip(self.backbone.encoder.encoder.layers,self.adapters):
            h=block(h)+adapter(h)
        h=self.read(h).permute(0,3,1,2)
        return self.head(F.adaptive_avg_pool2d(h,(self.pc,self.pt)))
    def delta(self):
        return {k:v.detach().cpu() for k,v in self.named_parameters() if v.requires_grad}
