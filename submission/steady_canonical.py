"""Canonical correlations to training-derived harmonic and evoked-response references."""
import math

import torch
from torch import nn

from steady_delay import SteadyDelay
from steady_grid import SteadyGrid


class CanonicalTemplates(nn.Module):
    def __init__(self, info):
        super().__init__()
        self.length = info["input_shapes"][0][1]
        self.fs = info["sampling_rate_hz"]
        self.short = self.length / self.fs < 1.5
        self.base = (SteadyGrid if self.short else SteadyDelay)(info)
        names = info["channel_names"]
        self.register_buffer("posterior", torch.tensor([i for i,n in enumerate(names)
                                                       if n.upper().startswith(("PO","O"))]))
        self.onsets = (0, round(.15*self.fs))
        self.bands = ((6, 88), (14, 88), (22, 88), (30, 88), (38, 88))
        freq = torch.fft.rfftfreq(self.length, 1/self.fs)
        self.register_buffer("filters", torch.stack([torch.sigmoid((freq-lo)*1.4)*torch.sigmoid((hi-freq)*1.4)
                                                    for lo,hi in self.bands]))
        for j, start in enumerate(self.onsets):
            self.register_buffer(f"refs{j}", torch.zeros(len(self.bands), self.base.k, 16, self.length-start))

    def bank(self, x, m):
        x = x[:, self.posterior] * m[:, self.posterior, None]
        x = x - x.mean(-1, keepdim=True)
        x = x / x.square().mean((1,2),keepdim=True).clamp_min(1e-10).sqrt()
        return torch.fft.irfft(torch.fft.rfft(x)[:,None] * self.filters[:,None], n=self.length)

    @torch.no_grad()
    def fit(self, xb, mb, labels, rows, device, groups):
        if self.short:
            self.base.fit(xb, mb, labels, rows, device, groups)
        else:
            self.base.fit(xb, mb, labels, rows, device)
        frequencies = self.base.reference_frequencies()
        counts = torch.zeros(self.base.k, device=device)
        mean = torch.zeros(self.base.k,len(self.bands),len(self.posterior),self.length,device=device)
        for i in range(0,len(rows),64):
            ids = rows[i:i+64]
            y = torch.as_tensor(labels[ids], device=device)
            bank = self.bank(torch.as_tensor(xb[ids],device=device), torch.as_tensor(mb[ids],device=device))
            mean.index_add_(0,y,bank)
            counts += torch.bincount(y,minlength=self.base.k)
        mean /= counts[:,None,None,None]
        mean -= mean.mean(0,keepdim=True)
        for j,start in enumerate(self.onsets):
            t = torch.arange(start,self.length,device=device)/self.fs
            phase = 2*math.pi*frequencies[:,None,None]*torch.arange(1,6,device=device)[:,None]*t
            harm = torch.cat((phase.cos(), phase.sin()),1)
            harm = torch.linalg.qr((harm-harm.mean(-1,keepdim=True)).mT).Q.mT.float()
            template = mean.permute(1,0,2,3)[...,start:]
            template = template-template.mean(-1,keepdim=True)
            _,_,basis = torch.linalg.svd(template,full_matrices=False)
            refs = getattr(self,f"refs{j}")
            refs[:,:,:10] = harm
            refs[:,:,10:] = basis[:,:,:6]
        return self

    def forward(self,x,m):
        parts = [self.base(x,m)]
        bank = self.bank(x,m)
        for j,start in enumerate(self.onsets):
            sig = bank[...,start:]
            sig = sig-sig.mean(-1,keepdim=True)
            cov = sig @ sig.mT
            ridge = .06*cov.diagonal(dim1=-1,dim2=-2).mean(-1).clamp_min(1e-6)
            eye = torch.eye(len(self.posterior),device=x.device)
            chol = torch.linalg.cholesky(cov + ridge[...,None,None]*eye)
            whitened = torch.linalg.solve_triangular(chol,sig,upper=False)
            cross = torch.einsum("bfct,fkht->bkfch",whitened,getattr(self,f"refs{j}"))
            for block in (cross[...,:10],cross[...,10:],cross[...,:2],cross[...,10:12]):
                gram = block @ block.mT
                eig = torch.linalg.eigvalsh(gram)
                parts.extend((eig[...,-1],eig[...,-2:].sum(-1),eig.sum(-1)))
        return torch.cat(parts,dim=-1)
