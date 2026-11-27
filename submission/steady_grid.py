"""Harmonic projections on a stimulation grid estimated from source EEG."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from steady_source import SteadySource


class SteadyGrid(SteadySource):
    def __init__(self, info):
        super().__init__(info)
        channels = len(self.sel)
        self.harmonics_count = 5
        samples = torch.arange(self.t)
        tapers = []
        for onset in (0, round(.14 * self.fs)):
            width = max(1, round(.05 * self.fs))
            rise = ((samples - onset) / width).clamp(0, 1)
            fall = ((self.t - 1 - samples) / width).clamp(0, 1)
            taper = torch.sin(math.pi * rise / 2).square() * torch.sin(math.pi * fall / 2).square()
            tapers.append(taper / taper.norm().clamp_min(1e-6))
        self.register_buffer('harmonic_tapers', torch.stack(tapers))
        self.register_buffer('grid_hz', torch.zeros(self.k))
        self.register_buffer('oscillators', torch.zeros(self.k, 2, 5, self.t, 2))
        self.register_buffer('mean_filter', torch.zeros(self.k, 2, 5, channels, 2))
        self.register_buffer('energy_filter', torch.zeros_like(self.mean_filter))
        self.register_buffer('harmonic_bias', torch.zeros(self.k, 2, 5, 2))
        self.register_buffer('latency_phase', torch.zeros(self.k, 5, 7, 2))

    def reference_frequencies(self):
        return self.grid_hz.double()

    def assign_grid(self, merit):
        # The training grid is already indexed by class.
        return np.arange(merit.shape[0])

    @torch.no_grad()
    def fit_grid(self, signals, masks, labels, rows, device):
        fft_size = 4000
        step = self.fs / fft_size
        cells = fft_size // 2 + 1
        power = torch.zeros(self.k, cells, dtype=torch.float64, device=device)
        means = torch.zeros(self.k, len(self.sel), self.t, dtype=torch.float64, device=device)
        counts = torch.zeros(self.k, dtype=torch.float64, device=device)
        for start in range(0, len(rows), 64):
            ix = rows[start:start + 64]
            x = self.raw(torch.as_tensor(signals[ix], device=device), torch.as_tensor(masks[ix], device=device))
            y = torch.as_tensor(labels[ix], device=device)
            tapered = x * self.harmonic_tapers[1]
            p = torch.fft.rfft(tapered, n=fft_size).abs().square().mean(1)
            power.index_add_(0, y, p.double())
            means.index_add_(0, y, tapered.double())
            counts += torch.bincount(y, minlength=self.k)
        if counts.min() == 0:
            raise ValueError('All stimulation classes are required in a source fold')
        power /= counts[:, None]
        means /= counts[:, None, None]
        coherent = torch.fft.rfft(means - means.mean(0), n=fft_size).abs().square().mean(1)
        evidence = (power / power.mean(0).clamp_min(1e-8)).clamp_min(.001).log()
        evidence += .25 * torch.log1p(coherent / coherent.mean(0).clamp_min(1e-8))
        label = torch.arange(self.k, device=device)
        harmonics = torch.arange(1, 6, device=device)
        origin = torch.arange(120, 211, device=device, dtype=torch.float64) / 20
        spacing = torch.arange(30, 51, device=device, dtype=torch.float64) / 200
        axes = torch.stack((label, label.flip(0)))
        grids = origin[:, None, None, None] + spacing[None, :, None, None] * axes
        indices = (grids[..., None] * harmonics / step).round().long().clamp(0, cells - 1)
        merits = evidence[label[:, None], indices].mean((-1, -2))
        grid = grids.reshape(-1, self.k)[merits.argmax()]
        self.grid_hz.copy_(grid)
        self.bins.copy_((grid[:, None] * harmonics * 2000 / self.fs).round().long())
        t = torch.arange(self.t, device=device) / self.fs
        phase = -2 * math.pi * self.grid_hz[:, None, None] * harmonics[:, None] * t
        osc = torch.polar(torch.ones_like(phase), phase)[:, None] * self.harmonic_tapers[None, :, None]
        self.oscillators.copy_(torch.view_as_real(osc.contiguous()))
        latencies = torch.linspace(-.048, .048, 7, device=device)
        delay = 2 * math.pi * self.grid_hz[:, None, None] * harmonics[:, None] * latencies
        self.latency_phase.copy_(torch.view_as_real(torch.polar(torch.ones_like(delay), delay)))

    def coefficients(self, x):
        c = x.shape[1]
        osc = self.oscillators.reshape(-1, self.t, 2)
        z = torch.complex(x @ osc[..., 0].T, x @ osc[..., 1].T)
        return z.reshape(len(x), c, self.k, 2, 5).permute(0, 2, 3, 4, 1)

    @torch.no_grad()
    def fit_harmonics(self, signals, masks, labels, rows, device):
        c = len(self.sel)
        shape = (self.k, 2, 5, c)
        signal = torch.zeros(shape, dtype=torch.complex128, device=device)
        mean = torch.zeros_like(signal)
        cov = torch.zeros((*shape, c), dtype=torch.complex128, device=device)
        target_cov = torch.zeros_like(cov)
        count = torch.zeros(self.k, dtype=torch.float64, device=device)
        for start in range(0, len(rows), 48):
            ix = rows[start:start + 48]
            x = self.raw(torch.as_tensor(signals[ix], device=device), torch.as_tensor(masks[ix], device=device))
            z = self.coefficients(x).to(torch.complex128)
            y = torch.as_tensor(labels[ix], device=device)
            matched = z[torch.arange(len(ix), device=device), y]
            signal.index_add_(0, y, matched)
            target_cov.index_add_(0, y, matched.unsqueeze(-1) * matched.conj().unsqueeze(-2))
            mean += z.sum(0)
            columns = z.permute(1, 2, 3, 4, 0)
            cov += columns @ columns.mH
            count += torch.bincount(y, minlength=self.k)
        mean /= len(rows)
        signal /= count[:, None, None, None]
        positive = target_cov / count[:, None, None, None, None]
        cov = cov / len(rows) - mean.unsqueeze(-1) * mean.conj().unsqueeze(-2)
        eye = torch.eye(c, device=device)
        scale = cov.diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(1e-6)
        eig, vec = torch.linalg.eigh(.82 * cov + .18 * scale[..., None, None] * eye)
        white = (vec * eig.clamp_min(1e-8).rsqrt().unsqueeze(-2)) @ vec.mH
        direction = white @ (signal - mean).unsqueeze(-1)
        direction /= direction.abs().square().sum(-2, keepdim=True).sqrt().clamp_min(1e-6)
        linear = (white @ direction).squeeze(-1)
        _, vec = torch.linalg.eigh(white @ positive @ white)
        quad = (white @ vec[..., -1:]).squeeze(-1)
        # The power filter is phase-free; fix its arbitrary complex sign for serialization.
        anchor = quad.gather(-1, quad.abs().argmax(-1, keepdim=True))
        quad = quad * (anchor.conj() / anchor.abs().clamp_min(1e-8))
        self.mean_filter.copy_(torch.view_as_real(linear.contiguous()))
        self.energy_filter.copy_(torch.view_as_real(quad.contiguous()))
        self.harmonic_bias.copy_(torch.view_as_real((mean * linear.conj()).sum(-1).contiguous()))

    @torch.no_grad()
    def fit(self, xb, mb, y, ids, device, groups):
        self.fit_grid(xb, mb, y, ids, device)
        super().fit(xb, mb, y, ids, device, groups)
        self.fit_harmonics(xb, mb, y, ids, device)
        return self

    def forward(self, x, mask):
        original = super().forward(x, mask)
        z = self.coefficients(self.raw(x, mask))
        v = (z * torch.view_as_complex(self.mean_filter).conj()).sum(-1)
        v = v - torch.view_as_complex(self.harmonic_bias)
        energy = (z * torch.view_as_complex(self.energy_filter).conj()).sum(-1).abs().square().log1p()
        unit = v / v.abs().clamp_min(.5)
        adjacent = unit[..., 1:] * unit[..., :-1].conj() * unit[..., :1].conj()
        phase = torch.view_as_complex(self.latency_phase)
        match = (v.unsqueeze(-1) * phase[:, None]).real
        parts = (energy, v.real, v.imag, v.abs().log1p(), adjacent.real, adjacent.imag,
                 match.sum(-2), match[..., :3, :].sum(-2))
        return torch.cat([original, *(p.flatten(2) for p in parts)], dim=-1)
