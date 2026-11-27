"""Offline OmniEEG single-window inference entry points."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from foundation import Foundation
from full import Full
from geometry import Geometry
from model import Features, Head
from montage import Montage
from motor import Motor
from motor_aligned import AlignedMotor, AlignedMontage
from neural import EEGAdapter, compact_public
from spectrogram import Spectrogram, SpectrumHead
from steady import Steady, SteadyHead, global_summary
from steady_delay import SteadyDelay
from steady_source import SteadySource
from steady_population import PopulationTemplates
from steady_canonical import CanonicalTemplates
from steady_window import SteadyWindow
from temporal import Temporal
from tree import evaluate
from tune import Tuned

FINE_TUNED = {"full": Full, "tune": Tuned, "montage": Montage, "motor_aligned": AlignedMotor,
              "montage_aligned": AlignedMontage}
SSVEP = {"steady": Steady, "window": SteadyWindow, "delay": SteadyDelay, "source": SteadySource, "evoked": PopulationTemplates, "canonical": CanonicalTemplates}
SPATIAL = {"geometry": Geometry, "temporal": Temporal, "motor": Motor}
SIGNAL_FIELDS = ("data", "x", "eeg", "signal", "signals", "inputs")


class Predictor:
    def __init__(
        self, root: Path, checkpoint: Path | dict, info: dict[str, Any], kind: str, device: str | torch.device
    ) -> None:
        self.info = info
        self.kind = kind
        self.device = torch.device(device)
        self.indices = list(range(len(info["label_values"])))
        self.back = self.features = self.geom = self.head = None
        ck = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location="cpu", weights_only=True)
        if kind == "fusion":
            if ck["kind"] != kind or ck["dataset"] != info["dataset"]:
                raise ValueError("Task and fusion checkpoint differ")
            self.factor = ck["factor"]
            self.parts = [Predictor(root, part["state"], info, part["architecture"], device)
                          for part in ck["parts"]]
            if info["dataset"] == "emotion_01":
                for part in self.parts:
                    part.float32_inference = True
            return
        weights = root / "assets" / "cbramod-pretrained.pth"

        if kind in FINE_TUNED:
            self.head = FINE_TUNED[kind](info, weights)
            params = dict(self.head.named_parameters())
            expected = {k for k, p in params.items() if p.requires_grad}
            if kind in ("motor_aligned", "montage_aligned"):
                expected.update(("source_center", "source_filter"))
                params.update(self.head.named_buffers())
            if set(ck["delta"]) != expected:
                raise ValueError("Incomplete fine-tuning weights")
            with torch.no_grad():
                for name, value in ck["delta"].items():
                    params[name].copy_(value)
        else:
            self.back = Foundation(info, weights)
            if kind == "stimulus_bag":
                self.features = Spectrogram(info)
                self.bag = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in ck["state"].items()}
            elif kind == "population":
                self.features = Spectrogram(info)
                self.population = {name: ck[name].to(device) for name in ("cols","mean","inv","coef")}
            elif kind == "rocket":
                from rocket_eeg import RocketFeatures
                self.features = RocketFeatures(info)
                self.rocket = {name: ck["state"][name].to(device) for name in ("center","scale","coef")}
                self.rocket_columns = [c.to(device) for c in ck["state"]["cols"]]
            elif kind == "kernel":
                self.features = Features(info)
                self.kernel = {name: value.to(device) for name, value in ck["state"].items()}
            elif kind.startswith("cbra_"):
                self.features = Features(info)
                self.head = Head(ck["dim"], len(self.indices), kind[5:])
                self.head.load_state_dict(ck["state"])
            elif kind == "spectrum":
                self.features = Spectrogram(info)
                self.head = SpectrumHead(info, ck["dim"], ck["bands"])
                self.head.load_state_dict(ck["state"])
            elif kind == "neural":
                self.head = EEGAdapter(info, ck["dim"])
                self.head.load_state_dict(ck["state"])
            elif kind in SSVEP:
                self.geom = SSVEP[kind](info)
                self.geom.load_state_dict(ck["proto"])
                head_type = SteadyHead
                if kind == "context":
                    from steady_context import FrequencyHead
                    head_type = FrequencyHead
                self.head = head_type(ck["dim"], 400, len(self.indices))
                self.head.load_state_dict(ck["head"])
            elif kind in SPATIAL:
                self.geom = SPATIAL[kind](info)
                self.geom.load_state_dict(ck["state"])
                self.mean = ck["mean"].to(device)
                self.inv = ck["inv"].to(device)
                self.coef = ck["coef"].to(device)
            elif kind == "tree":
                self.features = Features(info)
                self.cols = ck["cols"].to(device)
                self.tree = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in ck["tree"].items()}
            else:
                raise ValueError(f"Unknown architecture {kind}")

        for module in (self.back, self.features, self.geom, self.head):
            if module is not None:
                module.to(self.device).eval()

    @torch.inference_mode()
    def run(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.kind == "fusion":
            return self.parts[0].run(x, mask) + self.factor * self.parts[1].run(x, mask)
        if self.kind in ("full", "montage", "motor_aligned", "montage_aligned"):
            use_bf16 = (self.device.type == "cuda" and torch.cuda.is_bf16_supported()
                        and not getattr(self, "float32_inference", False))
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16):
                return self.head(x, mask).float()
        if self.kind == "tune":
            return self.head(x, mask)

        public = self.back(x, mask).half().float()
        if self.kind.startswith("cbra_"):
            return self.head(torch.cat((self.features(x, mask), public), dim=1))
        if self.kind in SSVEP:
            return self.head(self.geom(x, mask), global_summary(public, self.info))
        compact = compact_public(public, self.info)
        if self.kind == "stimulus_bag":
            from population import population_features
            from stimulus_bag import bag_scores
            features = population_features(self.features(x,mask).half(), mask, compact)
            score = bag_scores(features,self.bag) - self.bag["threshold"]
            return torch.stack((-score,score),dim=1)
        if self.kind == "population":
            from population import population_features
            features = population_features(self.features(x,mask).half(),mask,compact)[:, self.population["cols"]]
            z = ((features - self.population["mean"]) * self.population["inv"]).clamp(-8,8) @ self.population["coef"]
            return torch.stack((-z,z),dim=1)
        if self.kind == "rocket":
            selected = torch.cat((self.features(x,mask).half().float()[:,self.rocket_columns[0]],
                                  compact[:,self.rocket_columns[1]]),dim=1)
            return ((selected-self.rocket["center"])*self.rocket["scale"]) @ self.rocket["coef"]
        if self.kind == "kernel":
            from kernel_eeg import kernel_scores
            return kernel_scores(self.features(x, mask), compact, self.kernel)
        if self.kind == "spectrum":
            return self.head(self.features(x, mask).half(), compact)
        if self.kind == "neural":
            return self.head(x, mask, compact)
        if self.kind in SPATIAL:
            features = torch.cat((self.geom(x, mask), compact), dim=1)
            return ((features - self.mean) * self.inv).clamp(-8, 8) @ self.coef
        features = torch.cat((self.features(x, mask), compact), dim=1).index_select(1, self.cols)
        scores = evaluate(self.tree, features.cpu().numpy())
        return torch.as_tensor(scores, device=x.device, dtype=torch.float32)


def load_model(model_dir: str | Path, task_info: dict[str, Any], device: str | torch.device) -> Predictor:
    """Load the selected task and preserve the evaluator's requested class order."""
    directory = Path(model_dir)
    root = directory if (directory / "model_config.json").is_file() else directory.parent
    config = json.loads((root / "model_config.json").read_text())
    record = config["datasets"][task_info["dataset_name"]]
    source = record["info"]["label_values"]
    target = list(task_info.get("label_values", source))
    if sorted(source) != sorted(target):
        raise ValueError("Label set differs from the training release")
    torch.backends.mha.set_fastpath_enabled(False)
    model = Predictor(root, root / record["checkpoint"], record["info"], record["architecture"], device)
    model.indices = [source.index(label) for label in target]
    return model


def _tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)) and any(isinstance(item, torch.Tensor) for item in value):
        return torch.stack([torch.as_tensor(item, device=device) for item in value])
    return torch.as_tensor(np.asarray(value), device=device)


@torch.inference_mode()
def predict(model: Predictor, batch: Any) -> torch.Tensor:
    """Return a finite float32 score matrix with one independent row per input window."""
    channels, length = model.info["input_shapes"][0]
    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
        key = next(k for k in SIGNAL_FIELDS if k in batch[0])
        signal = [row[key] for row in batch]
        mask = [row.get("channel_mask") for row in batch]
    elif isinstance(batch, dict):
        key = next(k for k in SIGNAL_FIELDS if k in batch)
        signal, mask = batch[key], batch.get("channel_mask")
    else:
        signal, mask = batch, None

    x = _tensor(signal, model.device).float()
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim == 3 and x.shape[1:] == (length, channels) and length != channels:
        x = x.transpose(1, 2)
    if x.ndim == 2 and tuple(x.shape) == (channels, length):
        x = x[None]
    elif x.ndim == 1 or (x.ndim == 2 and x.shape[1] == channels * length):
        x = x.reshape(-1, channels, length)
    if x.ndim != 3 or tuple(x.shape[1:]) != (channels, length):
        raise ValueError(f"Unexpected EEG shape {tuple(x.shape)}")

    if isinstance(mask, (tuple, list)) and any(item is None for item in mask):
        mask = [np.ones(channels, dtype=bool) if item is None else item for item in mask]
    if mask is None:
        valid = torch.ones((len(x), channels), dtype=torch.bool, device=x.device)
    else:
        valid = _tensor(mask, x.device).bool().reshape(-1, channels)
    if tuple(valid.shape) != (len(x), channels):
        raise ValueError("Channel mask shape mismatch")

    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    results = []
    fixed_shape = model.kind == "fusion" and model.info["dataset"] == "emotion_01"
    size = 8 if fixed_shape else 24
    for i in range(0, len(x), size):
        signal_rows, mask_rows = x[i:i+size], valid[i:i+size]
        count = len(signal_rows)
        if fixed_shape and count < size:
            signal_rows = torch.cat((signal_rows, signal_rows[:1].expand(size-count,-1,-1)), dim=0)
            mask_rows = torch.cat((mask_rows, mask_rows[:1].expand(size-count,-1)), dim=0)
        results.append(model.run(signal_rows, mask_rows)[:count])
    if not results:
        return torch.empty((0, len(model.indices)), dtype=torch.float32)
    output = torch.cat(results, 0)[:, model.indices].float().cpu()
    if not torch.isfinite(output).all():
        raise ValueError("Nonfinite prediction")
    return output
