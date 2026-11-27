"""One shared classifier over source-trained steady-state response descriptors."""
import torch
from torch import nn

from steady_canonical import CanonicalTemplates
from steady_population import PopulationTemplates


class ResponseDescriptors(nn.Module):
    def __init__(self, info):
        super().__init__()
        self.canonical = CanonicalTemplates(info)
        self.population = PopulationTemplates(info)

    def fit(self, signal, masks, y, ids, device, groups):
        self.canonical.fit(signal, masks, y, ids, device, groups)
        self.population.fit(signal, masks, y, ids, device, groups)
        return self

    def forward(self, signal, mask):
        return torch.cat((self.canonical(signal, mask), self.population(signal, mask)), dim=-1)


class ResponseHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('center',torch.zeros(dim))
        self.register_buffer('scale',torch.ones(dim))
        self.layers = nn.Sequential(nn.Dropout(.12), nn.Linear(dim*2,64), nn.LayerNorm(64),
                                    nn.GELU(), nn.Dropout(.15), nn.Linear(64,1))

    def forward(self, features):
        z = ((features - self.center) * self.scale).clamp(-8,8)
        relative = (z - z.mean(1,keepdim=True)) / z.std(1,unbiased=False,keepdim=True).clamp_min(.30)
        return self.layers(torch.cat((z,relative.clamp(-8,8)), dim=-1)).squeeze(-1)
