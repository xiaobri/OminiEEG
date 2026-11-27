"""Harmonic response subspaces fitted to source subjects' evoked waveforms."""
import numpy as np
import torch
from torch.nn import functional as F
from steady_delay import SteadyDelay


class SteadySource(SteadyDelay):
    def __init__(self, info):
        super().__init__(info)
        self.components = 4
        self.register_buffer('subject_basis',torch.zeros(5,self.k,self.components,self.delay_rank,10))
        self.register_buffer('subject_mean',torch.zeros(5,self.k,self.delay_rank,10))

    @torch.no_grad()
    def fit(self, xb, mb, y, ids, device, groups):
        super().fit(xb,mb,y,ids,device)
        subject_means = []
        for subject in np.unique(groups[ids]):
            rows = ids[groups[ids]==subject]
            waves = torch.zeros(5,self.k,self.delay_rank,self.length,device=device)
            counts = torch.zeros(self.k,device=device)
            for start in range(0,len(rows),64):
                ix = rows[start:start+64]
                x = self.raw(torch.as_tensor(xb[ix],device=device),torch.as_tensor(mb[ix],device=device))
                h = torch.einsum('bfct,fcr->bfrt',self.embedded(x),self.delay_basis)
                label = torch.as_tensor(y[ix],device=device)
                waves.index_add_(1,label,h.transpose(0,1))
                counts += torch.bincount(label,minlength=self.k)
            valid = counts>0
            waves /= counts.clamp_min(1)[None,:,None,None]
            waves -= waves[:,valid].mean(1,keepdim=True)
            weights = counts / (counts+2.)
            coef = (waves @ self.references.mT).flatten(-2)
            subject_means.append(F.normalize(coef,dim=-1)*weights[None,:,None])
        data = torch.stack(subject_means,dim=2)
        mean = F.normalize(data.mean(2),dim=-1)
        self.subject_mean.copy_(mean.reshape_as(self.subject_mean))
        # A single class subspace captures reproducible harmonic phase variation.
        resid = data - (data*mean[:,:,None]).sum(-1,keepdim=True)*mean[:,:,None]
        _,vectors = torch.linalg.eigh(resid @ resid.mT)
        k = min(self.components,len(subject_means))
        basis = F.normalize(vectors[:,:,:,-k:].mT @ resid,dim=-1)
        self.subject_basis.zero_()
        self.subject_basis[:,:,:k].copy_(basis.reshape(5,self.k,k,self.delay_rank,10))
        return self

    def forward(self,x,mask):
        original = super().forward(x,mask)
        h = torch.einsum('bfct,fcr->bfrt',self.embedded(self.raw(x,mask)),self.delay_basis)
        h = F.normalize(h.flatten(-2),dim=-1).reshape_as(h)
        coef = torch.einsum('bfrt,kht->bfkrh',h,self.references)
        parts = [original]
        for scale in (coef, F.normalize(coef.flatten(-2),dim=-1).reshape_as(coef)):
            match = (scale * self.subject_mean).sum((-1,-2))
            variability = torch.einsum('bfkrh,fkprh->bfkp',scale,self.subject_basis).square()
            parts.append(match.transpose(1,2))
            parts.append(variability.cumsum(-1).clamp_min(1e-12).sqrt().transpose(1,2).flatten(2))
        return torch.cat(parts,dim=-1)
