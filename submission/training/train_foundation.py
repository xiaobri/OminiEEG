import argparse,json,os
import numpy as np
import torch
from foundation import Foundation,MatrixPair
from train_features import fit
from splits import ROOT

def encoded(task,device):
    base=ROOT/'cache'/task
    dest=base/'foundation.npy'
    if not dest.is_file():
        info=json.loads((base/'info.json').read_text());x=np.load(base/'x.npy',mmap_mode='r');m=np.load(base/'mask.npy',mmap_mode='r')
        model=Foundation(info,ROOT/'assets/cbramod-pretrained.pth').eval().to(device)
        result=None
        with torch.inference_mode():
            for at in range(0,len(x),32):
                xx=torch.from_numpy(x[at:at+32].copy()).to(device);mm=torch.from_numpy(m[at:at+32].copy()).to(device)
                v=model(xx,mm).float().cpu().numpy().astype('float16')
                if not np.isfinite(v).all():raise ValueError('nonfinite public encoder output')
                if result is None:result=np.lib.format.open_memmap(dest.with_suffix('.tmp'),mode='w+',dtype='float16',shape=(len(x),v.shape[1]))
                result[at:at+len(v)]=v
        result.flush();del result;os.replace(dest.with_suffix('.tmp'),dest)
    return dest

def develop(task,device):
    base=ROOT/'cache'/task;path=encoded(task,device)
    public=np.load(path,mmap_mode='r')
    v=MatrixPair(np.load(base/'features-v1.npy',mmap_mode='r'),public)
    y=np.load(base/'y.npy');g=np.load(base/'groups.npy');pairs=json.loads((base/'splits.json').read_text())
    for fold,group in enumerate(pairs['dev_folds']):
        tst=np.isin(g,group);tr=np.flatnonzero(~np.isin(g,pairs['blind'])&~tst);va=np.flatnonzero(tst)
        for kind in ['mlp','linear']:
            out=ROOT/'runs/development'/task/('cbra_'+kind)/f'fold{fold}'
            if (out/'metrics.json').is_file():continue
            score=fit(v,y,tr,va,kind,device,20,out,4400+fold)
            print(task,'public encoder',kind,fold,round(score,4),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task');a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.22)
    torch.backends.mha.set_fastpath_enabled(False)
    develop(a.task,'cuda')
