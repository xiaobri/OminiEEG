import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(os.environ['OMNIEEG_TRAIN_WORKDIR']).resolve() if 'OMNIEEG_TRAIN_WORKDIR' in os.environ else Path(__file__).resolve().parents[1]

def split_groups(y,g,indices,k):
    counts=np.bincount(y[indices])
    local_y,local_g=y[indices],g[indices]
    ideal=len(indices)/k
    best=None
    for seed in (232,29,133,57,61,71,92,123,181,210,1001,2026):
        parts=[indices[v] for _,v in StratifiedGroupKFold(k,shuffle=True,random_state=seed).split(local_y,local_y,local_g)]
        ratios=np.stack([np.bincount(y[p],minlength=len(counts))/counts for p in parts])
        score=np.max(np.abs(ratios*k-1))+np.std(ratios*k)+0.1*max(abs(len(p)/ideal-1) for p in parts)
        if best is None or score<best[0]:best=(score,parts)
    return best[1]

def plan(base):
    y=np.load(base/'y.npy');g=np.load(base/'groups.npy')
    ix=np.arange(len(y));ng=len(np.unique(g))
    outer=split_groups(y,g,ix,min(5,ng))
    blind=outer[0];dev=np.setdiff1d(ix,blind)
    inner=split_groups(y,g,dev,min(3,len(np.unique(g[dev]))))
    vv=[np.unique(g[p]).tolist() for p in inner]
    assert len(set.intersection(set(np.unique(g[blind])),set(np.unique(g[dev]))))==0
    assert all(len(np.unique(y[p]))==len(np.unique(y)) for p in [blind,*inner])
    return dict(blind=np.unique(g[blind]).tolist(),dev_folds=vv)
