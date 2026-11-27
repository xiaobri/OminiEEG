"""Refit both source-trained branches and serialize a single task-bound checkpoint."""
import json
from pathlib import Path

import torch


def pack(task, rec, sources, target):
    parts = [dict(architecture=spec["architecture"], state=torch.load(path, map_location="cpu", weights_only=True))
             for spec, path in zip(rec["parts"], sources, strict=True)]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    torch.save(dict(kind="fusion", dataset=task, factor=rec["factor"], parts=parts), tmp)
    tmp.replace(target)


def fit_fusion(task, rec, tr, out, seed, device):
    from finalize_models import refit
    sources = []
    for i, part in enumerate(rec["parts"]):
        directory = out / f"branch{i}"
        refit(task, part, tr, directory, seed, device)
        sources.append(directory / "model.pt")
    pack(task, rec, sources, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(dict(rows=len(tr), seed=seed)))
