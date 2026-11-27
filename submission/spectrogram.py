"""Single-window spectrogram adapter for a frozen public CBraMod backbone."""
import re

import torch
from torch import nn
from torch.nn import functional as F


class Spectrogram(nn.Module):
    def __init__(self, info):
        super().__init__()
        channels, length = info['input_shapes'][0]
        fs = info['sampling_rate_hz']
        self.size = min(length, round(fs))
        self.step = max(1, self.size // 4)
        self.frames = min(24, 1 + (length - self.size) // self.step)
        freq = torch.fft.rfftfreq(self.size, 1 / fs)
        bands = [(0.5, 3), (3, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 19), (19, 23),
                 (23, 27), (27, 32), (32, 39), (39, 46), (46, 55), (55, 70), (70, 95), (95, 140), (140, 240), (240, 450)]
        bins = torch.stack([(freq >= lo) & (freq < hi) for lo, hi in bands], 1).float()
        bins = bins[:, bins.sum(0) > 0]
        self.bands = bins.shape[1]
        self.register_buffer('bands_matrix', bins / bins.sum(0))
        self.register_buffer('window', torch.hann_window(self.size))

    def forward(self, raw, mask):
        x = raw.float() * mask[:, :, None]
        x = x - x.mean(-1, keepdim=True)
        win = x.unfold(-1, self.size, self.step) * self.window
        power = torch.fft.rfft(win).abs().square() / self.size @ self.bands_matrix
        logp = power.clamp_min(1e-10).log().transpose(-1, -2)
        return F.adaptive_avg_pool1d(logp.flatten(1, 2), self.frames).reshape(len(x), x.shape[1], self.bands, self.frames)


class SpectrumHead(nn.Module):
    def __init__(self, info, public_dim, bands):
        super().__init__()
        channels = info['input_shapes'][0][0]
        self.register_buffer('mean', torch.zeros(1,channels,bands,1))
        self.register_buffer('scale', torch.ones_like(self.mean))
        self.register_buffer('pm', torch.zeros(public_dim))
        self.register_buffer('ps', torch.ones(public_dim))
        names = [str(name).strip('.').upper() for name in info['channel_names']]
        pairs = []
        for i, name in enumerate(names):
            match = re.fullmatch(r'([A-Z]+)([13579])', name)
            if match:
                counterpart = match[1] + str(int(match[2]) + 1)
                if counterpart in names:
                    pairs.append((i, names.index(counterpart)))
        self.register_buffer('left', torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer('right', torch.tensor([p[1] for p in pairs], dtype=torch.long))
        self.project = nn.Sequential(nn.Conv2d(channels + len(pairs),64,1,bias=False),nn.BatchNorm2d(64),nn.GELU(),
                                     nn.Conv2d(64,64,(3,3),padding=1,groups=64,bias=False),nn.BatchNorm2d(64),nn.GELU(),
                                     nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.GELU())
        self.local = nn.Sequential(nn.Flatten(),nn.Dropout(.3),nn.Linear(64*4*3,144),nn.LayerNorm(144),nn.GELU())
        self.public = nn.Sequential(nn.Dropout(.3),nn.Linear(public_dim,80),nn.LayerNorm(80),nn.GELU())
        self.head = nn.Sequential(nn.Dropout(.35), nn.Linear(224,len(info['label_values'])))

    def forward(self, spec, pub):
        x = ((spec.float() - self.mean)*self.scale).clamp(-8,8)
        if self.training:
            # Source-only random electrode gain and smooth broadband spectral offsets.
            x = x + torch.randn_like(x[:,:,:1,:1])*.16 + torch.randn_like(x[:,:1,:,:1])*.10
        delta = x[:, self.left] - x[:, self.right]
        h = self.project(torch.cat((x,delta),dim=1))
        h = self.local(F.adaptive_avg_pool2d(h,(4,3)))
        p = self.public(((pub-self.pm)*self.ps).clamp(-8,8))
        return self.head(torch.cat((h,p),dim=1))
