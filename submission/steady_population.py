"""Spatial whitening and cross-subject evoked-response subspaces for SSVEP."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PopulationTemplates(nn.Module):
    def __init__(self, info):
        super().__init__()
        self.classes = len(info["label_values"])
        _, self.length = info["input_shapes"][0]
        channels = [i for i, name in enumerate(info["channel_names"])
                    if name.upper().startswith(("P", "O", "CP", "TP"))]
        self.register_buffer("channels", torch.tensor(channels))
        fs = info["sampling_rate_hz"]
        freq = torch.fft.rfftfreq(self.length, 1 / fs)
        bands = [(6, 18), (14, 34), (22, 48), (31, 65), (42, 85)]
        filters = torch.stack([torch.sigmoid((freq-lo)*1.5) * torch.sigmoid((hi-freq)*1.5) for lo, hi in bands])
        self.register_buffer("filters", filters)
        self.register_buffer("transform", torch.zeros(len(bands), len(channels), 6))
        # Eight directions per class describe the mean response and its source-subject variation.
        self.register_buffer("templates", torch.zeros(len(bands), self.classes, 8, 6, self.length))
        self.register_buffer("strength", torch.zeros(len(bands), self.classes, 8))
        self.register_buffer("frequency_scale", torch.ones(len(freq)))

    def spectrum(self, x, mask):
        x = x[:, self.channels].float() * mask[:, self.channels, None]
        x = x - x.mean(-1, keepdim=True)
        x = x / x.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
        return torch.fft.rfft(x, norm="ortho")

    def bank(self, x, mask):
        spec = self.spectrum(x, mask) * self.frequency_scale
        return torch.fft.irfft(spec[:, None] * self.filters[:, None], n=self.length, norm="ortho")

    @torch.no_grad()
    def fit(self, xb, mb, labels, rows, device, groups):
        subjects, inverse = np.unique(groups[rows], return_inverse=True)
        count = torch.zeros((len(subjects), self.classes), device=device)
        shape = (len(subjects), self.classes, len(self.channels), self.length)
        means = torch.zeros(shape, device=device)
        spectrum = torch.zeros_like(self.frequency_scale)
        for i in range(0, len(rows), 64):
            ids = rows[i:i+64]
            sub = torch.as_tensor(inverse[i:i+len(ids)], device=device)
            y = torch.as_tensor(labels[ids], device=device)
            z = self.spectrum(torch.as_tensor(xb[ids], device=device), torch.as_tensor(mb[ids], device=device))
            spectrum += z.abs().square().sum((0, 1)) / (len(rows)*len(self.channels))
            means.flatten(0, 1).index_add_(0, sub*self.classes+y, torch.fft.irfft(z, n=self.length, norm="ortho"))
            count.view(-1).index_add_(0, sub*self.classes+y, torch.ones_like(y, dtype=torch.float32))
        if (count == 0).any():
            raise ValueError("Expected repeated presentations of every class for each source subject")
        self.frequency_scale.copy_(spectrum.clamp_min(.003).rsqrt().clamp_max(12))
        means /= count[:, :, None, None]
        means -= means.mean(1, keepdim=True)
        spectrum_means = torch.fft.rfft(means, norm="ortho") * self.frequency_scale
        covariance = torch.zeros(len(self.filters), len(self.channels), len(self.channels), dtype=torch.float64, device=device)
        for i in range(0, len(rows), 64):
            ids = rows[i:i+64]
            bank = self.bank(torch.as_tensor(xb[ids], device=device), torch.as_tensor(mb[ids], device=device))
            covariance += (bank @ bank.mT).sum(0).double() / (self.length * len(rows))
        for f, filt in enumerate(self.filters):
            wave = torch.fft.irfft(spectrum_means * filt, n=self.length, norm="ortho")
            ev, vec = torch.linalg.eigh(.97 * covariance[f] + .03 * covariance[f].diagonal().mean() *
                                        torch.eye(len(self.channels), device=device))
            white = (vec * ev.clamp_min(1e-7).rsqrt()) @ vec.T
            avg = wave.mean(0)
            # Discriminant spatial directions reward signals shared between people.
            shared = (avg @ avg.mT).mean(0).double()
            _, directions = torch.linalg.eigh(white @ shared @ white)
            basis = (white @ directions[:, -6:]).float()
            self.transform[f].copy_(basis)
            projected = (basis.T @ wave).transpose(0, 1).flatten(-2)
            projected = F.normalize(projected, dim=-1)
            mean = F.normalize(projected.mean(1), dim=-1)
            residual = projected - (projected * mean[:, None]).sum(-1, keepdim=True) * mean[:, None]
            ev, vec = torch.linalg.eigh(residual @ residual.mT)
            n = min(7, len(subjects)-1)
            pcs = F.normalize(vec[:, :, -n:].flip(-1).mT @ residual, dim=-1)
            self.templates[f, :, 0].copy_(mean.reshape(self.classes, 6, self.length))
            self.templates[f, :, 1:1+n].copy_(pcs.reshape(self.classes, n, 6, self.length))
            self.strength[f, :, 0] = 1
            self.strength[f, :, 1:1+n] = ev[:, -n:].flip(-1).clamp_min(0).sqrt() / len(subjects)**.5
        return self

    def forward(self, x, mask):
        bank = self.transform.mT @ self.bank(x, mask)
        norm = F.normalize(bank.flatten(-2), dim=-1)
        refs = self.templates.flatten(-2)
        parts = []
        for lag in (-4, 0, 4):
            signal = norm if lag == 0 else F.normalize(bank.roll(lag, -1).flatten(-2), dim=-1)
            match = torch.einsum("bfd,fkpd->bkfp", signal, refs)
            variation = match[..., 1:].square() * self.strength.transpose(0, 1)[..., 1:]
            parts.extend((match[..., 0], variation[..., :2].sum(-1), variation.sum(-1)))
        return torch.cat(parts, dim=-1)
