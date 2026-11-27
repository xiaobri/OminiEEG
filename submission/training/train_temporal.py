import json
import sys

import numpy as np
import torch

from splits import ROOT
from train_geometry import fit_geometry


def fit_temporal(task, tr, va, out, device="cuda", epochs=1, seed=169):
    return fit_geometry(task, tr, va, out, device, epochs, seed, kind="temporal")


if __name__ == "__main__":
    task = sys.argv[1]
    base = ROOT / "cache" / task
    groups = np.load(base / "groups.npy")
    splits = json.loads((base / "splits.json").read_text())
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.22)
    dev = ~np.isin(groups, splits["blind"])
    for fold, subjects in enumerate(splits["dev_folds"]):
        out = ROOT / "runs/development" / task / "temporal" / f"fold{fold}"
        if (out / "metrics.json").is_file():
            continue
        held = np.isin(groups, subjects)
        print(task, fold, fit_temporal(task, np.flatnonzero(dev & ~held), np.flatnonzero(held), out), flush=True)
