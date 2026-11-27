"""Source-only spatial whitening for the motor imagery CBraMod adapter."""
import numpy as np
import torch

from full import Full


class AlignedMotor(Full):
    checkpoint_kind = 'motor_aligned'

    def __init__(self, info, weights):
        super().__init__(info, weights)
        channels = info['input_shapes'][0][0]
        self.register_buffer('source_center', torch.zeros(channels, 1))
        self.register_buffer('source_filter', torch.eye(channels))

    @torch.no_grad()
    def fit_source(self, x, mask, groups, rows, device):
        """Average the training subjects' covariances with equal weight."""
        subjects = np.unique(groups[rows])
        channels = self.source_filter.shape[0]
        total = torch.zeros((channels, channels), dtype=torch.float64, device=device)
        level = torch.zeros(channels, dtype=torch.float64, device=device)
        for subject in subjects:
            indices = rows[groups[rows] == subject]
            sums = torch.zeros_like(total)
            pairs = torch.zeros_like(total)
            means = torch.zeros_like(level)
            counts = torch.zeros_like(level)
            for start in range(0, len(indices), 64):
                batch = indices[start:start + 64]
                signal = torch.as_tensor(x[batch], device=device, dtype=torch.float64)
                valid = torch.as_tensor(mask[batch], device=device, dtype=torch.float64)
                signal = torch.nan_to_num(signal, nan=0., posinf=0., neginf=0.)
                means += (signal.mean(-1) * valid).sum(0)
                counts += valid.sum(0)
                centered = (signal - signal.mean(-1, keepdim=True)) * valid.unsqueeze(-1)
                sums += (centered @ centered.mT).sum(0) / signal.shape[-1]
                pairs += valid.T @ valid
            cov = sums / pairs.clamp_min(1)
            scale = cov.diag().sum() / (counts > 0).sum().clamp_min(1)
            total += cov / scale.clamp_min(1e-8) / len(subjects)
            level += means / counts.clamp_min(1) / len(subjects)
        eye = torch.eye(channels, device=device, dtype=total.dtype)
        eig, vec = torch.linalg.eigh(.75 * total + .25 * eye)
        self.source_filter.copy_((vec * eig.clamp_min(.25).pow(-.25).unsqueeze(0)) @ vec.T)
        self.source_center.copy_(level[:, None])

    def forward(self, x, mask):
        with torch.autocast(device_type=x.device.type, enabled=False):
            clean = torch.nan_to_num(x.float(), nan=0., posinf=0., neginf=0.)
            filtered = self.source_filter @ ((clean - self.source_center) * mask.unsqueeze(-1))
        return super().forward(filtered, mask)

    def delta(self):
        state = super().delta()
        state['source_filter'] = self.source_filter.detach().cpu()
        state['source_center'] = self.source_center.detach().cpu()
        return state
