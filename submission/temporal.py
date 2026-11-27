"""Training-subject spatial filters for evoked responses and time-varying power."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Temporal(nn.Module):
    def __init__(self, info):
        super().__init__()
        channels, length = info["input_shapes"][0]
        self.classes = len(info["label_values"])
        self.rank = min(12, channels)
        frequency = torch.fft.rfftfreq(length, 1 / info["sampling_rate_hz"])
        bands = [(0.7, 8), (4, 15), (8, 30), (20, 50)]
        self.register_buffer("response", torch.stack([
            torch.sigmoid((frequency - lo) * 1.5) * torch.sigmoid((hi - frequency) * 1.5)
            for lo, hi in bands
        ]))
        self.register_buffer("basis", torch.zeros(len(bands), channels, self.rank))

    def bank(self, raw, mask):
        x = raw.float() * mask.unsqueeze(-1)
        x = x - x.mean(-1, keepdim=True)
        x = x / x.square().mean((1, 2), keepdim=True).clamp_min(1e-10).sqrt()
        return torch.fft.irfft(torch.fft.rfft(x)[:, None] * self.response[:, None], n=x.shape[-1])

    @torch.no_grad()
    def fit(self, raw, mask, labels, groups, rows, device, seed):
        rng = np.random.default_rng(seed)
        bands, channels, rank = self.basis.shape
        length = raw.shape[-1]
        means = torch.zeros(self.classes, bands, channels, length, device=device, dtype=torch.float64)
        cov = torch.zeros(bands, channels, channels, device=device, dtype=torch.float64)
        for label in range(self.classes):
            examples = rows[labels[rows] == label]
            subjects = np.unique(groups[examples])
            for subject in subjects:
                ix = examples[groups[examples] == subject]
                ix = rng.choice(ix, min(len(ix), max(16, min(240, 2500 // len(subjects)))), replace=False)
                weight = 1.0 / (len(ix) * len(subjects))
                for batch in np.array_split(ix, (len(ix) + 39) // 40):
                    wave = self.bank(torch.as_tensor(raw[batch], device=device), torch.as_tensor(mask[batch], device=device))
                    means[label] += wave.sum(0, dtype=torch.float64) * weight
                    cov += torch.einsum('bfct,bfdt->fcd', wave, wave).double() * (weight / (length * self.classes))
        means -= means.mean(0, keepdim=True)
        signal = torch.einsum("kfct,kfdt->fcd", means, means) / (length * self.classes)
        diag = cov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
        eig, vec = torch.linalg.eigh(.85 * cov + .15 * diag * torch.eye(channels, device=device))
        white = (vec * eig.clamp_min(1e-8).rsqrt().unsqueeze(-2)) @ vec.mT
        _, vec = torch.linalg.eigh(white @ signal @ white)
        self.basis.copy_(white @ vec[:, :, -rank:])
        return self

    def forward(self, raw, mask):
        v = torch.einsum("bfct,fcr->bfrt", self.bank(raw, mask), self.basis).flatten(1, 2)
        wave = F.adaptive_avg_pool1d(v, 32)
        power = F.adaptive_avg_pool1d(v.square(), 8).clamp_min(1e-8).log()
        return torch.cat((wave.flatten(1), power.flatten(1)), 1)
