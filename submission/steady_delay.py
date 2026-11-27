"""Source-only delay-embedded discriminant filters and harmonic templates for SSVEP."""
import math

import torch
from torch.nn import functional as F

from steady_window import SteadyWindow


class SteadyDelay(SteadyWindow):
    def __init__(self, info):
        super().__init__(info)
        self.lags = (0, 2, 5, 9)
        self.start = round(.12 * info["sampling_rate_hz"])
        self.fs = info["sampling_rate_hz"]
        self.length = self.t - self.start - max(self.lags)
        self.delay_rank = 10
        n = len(self.sel) * len(self.lags)
        self.register_buffer("delay_basis", torch.zeros(5, n, self.delay_rank))
        self.register_buffer("delay_template", torch.zeros(5, self.k, self.delay_rank, self.length))
        self.register_buffer("harmonic_template", torch.zeros_like(self.delay_template))
        self.register_buffer("references", torch.zeros(self.k, 10, self.length))

    def embedded(self, raw):
        filtered = self.bank(raw)
        return torch.cat([filtered[..., self.start+d:self.start+d+self.length] for d in self.lags], dim=2)

    def reference_frequencies(self):
        return 8.0 + self.freq_order.double() * .2

    @torch.no_grad()
    def fit(self, xb, mb, y, ids, device):
        super().fit(xb, mb, y, ids, device)
        bands, dims, rank = self.delay_basis.shape
        cov = torch.zeros(bands, dims, dims, device=device, dtype=torch.float64)
        templates = torch.zeros(bands, self.k, dims, self.length, device=device, dtype=torch.float64)
        counts = torch.zeros(self.k, device=device, dtype=torch.float64)
        for start in range(0, len(ids), 48):
            ix = ids[start:start+48]
            x = self.raw(torch.as_tensor(xb[ix], device=device), torch.as_tensor(mb[ix], device=device))
            bank = self.embedded(x)
            labels = torch.as_tensor(y[ix], device=device)
            cov += torch.einsum("bfct,bfdt->fcd", bank, bank).double()
            templates.index_add_(1, labels, bank.transpose(0, 1).double())
            counts += torch.bincount(labels, minlength=self.k)
        templates /= counts[None, :, None, None]
        templates -= templates.mean(1, keepdim=True)
        cov /= len(ids) * self.length
        signal = torch.einsum("fkct,fkdt->fcd", templates, templates) / (self.k * self.length)
        eig, vec = torch.linalg.eigh(.94 * cov + .06 * torch.diag_embed(cov.diagonal(dim1=-2, dim2=-1)))
        white = (vec * eig.clamp_min(1e-8).rsqrt().unsqueeze(-2)) @ vec.mT
        _, vec = torch.linalg.eigh(white @ signal @ white)
        basis = white @ vec[:, :, -rank:]
        self.delay_basis.copy_(basis)
        self.delay_template.copy_(torch.einsum("fcr,fkct->fkrt", basis, templates))
        freqs = self.reference_frequencies()[:, None, None]
        phase = 2 * math.pi * freqs * torch.arange(1, 6, device=device)[None, :, None] * torch.arange(self.length, device=device) / self.fs
        q, _ = torch.linalg.qr(torch.cat((phase.sin(), phase.cos()), dim=1).mT)
        self.references.copy_(q.mT)
        self.harmonic_template.copy_((self.delay_template @ self.references.mT) @ self.references)
        return self

    def forward(self, x, mask):
        original = super().forward(x, mask)
        bank = self.embedded(self.raw(x, mask))
        features = torch.einsum("bfct,fcr->bfrt", bank, self.delay_basis)
        pieces = [original]
        for trim in (0, self.length // 4):
            for rank in (3, self.delay_rank):
                projected = F.normalize(features[:, :, -rank:, trim:].flatten(-2), dim=-1)
                for template in (self.delay_template, self.harmonic_template):
                    ref = template[:, :, -rank:, trim:]
                    correlations = []
                    for delta in (-3, 0, 3):
                        vec = F.normalize(ref.roll(delta, dims=-1).flatten(-2), dim=-1)
                        correlations.append(torch.einsum("bfn,fkn->bkf", projected, vec))
                    corr = torch.stack(correlations, -1)
                    pieces.extend((corr.flatten(2), corr.amax(-1)))
        wave = F.normalize(features, dim=-1)
        harmonic = torch.einsum("bfrt,kht->bkfrh", wave, self.references).square().sum(-1)
        pieces.append(harmonic.mean(-1))
        return torch.cat(pieces, dim=-1)
