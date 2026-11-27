import json,os
from pathlib import Path
import numpy as np
import torch
from model import Features
from splits import ROOT,plan

def make(task,device):
    base=ROOT/'cache'/task
    info=json.loads((base/'info.json').read_text())
    if not (base/'splits.json').exists():
        tmp=base/'splits.tmp';tmp.write_text(json.dumps(plan(base)));os.replace(tmp,base/'splits.json')
    out=base/'features-v1.npy'
    if out.is_file():return
    x=np.load(base/'x.npy',mmap_mode='r');m=np.load(base/'mask.npy',mmap_mode='r')
    extractor=Features(info).to(device).eval(); dest=None
    with torch.inference_mode():
        for start in range(0,len(x),96):
            stop=min(start+96,len(x))
            v=extractor(torch.from_numpy(x[start:stop].copy()).to(device),torch.from_numpy(m[start:stop].copy()).to(device)).cpu().numpy()
            assert np.isfinite(v).all()
            if dest is None:dest=np.lib.format.open_memmap(out.with_suffix('.tmp'),mode='w+',dtype='float32',shape=(len(x),v.shape[1]))
            dest[start:stop]=v
    dest.flush();del dest;os.replace(out.with_suffix('.tmp'),out)
