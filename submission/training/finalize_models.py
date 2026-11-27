"""Independent grouped holdout assessment and full public-data refits."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from splits import ROOT
from foundation import MatrixPair
from train_features import fit
from predict import Predictor

SEEDS=[431,1907,8849]

def public_info(task):
    info=json.loads((ROOT/'cache'/task/'info.json').read_text())
    return {k:info[k] for k in ('dataset','input_shapes','sampling_rate_hz','label_values','channel_names')}

def refit(task,rec,tr,out,seed,device,validation=None):
    k=rec['architecture'];ep=rec['epochs'];va=np.zeros(0,dtype='int64') if validation is None else validation
    b=ROOT/'cache'/task
    if k.startswith('cbra_'):
        x=MatrixPair(np.load(b/'features-v1.npy',mmap_mode='r'),np.load(b/'foundation.npy',mmap_mode='r'))
        return fit(x,np.load(b/'y.npy'),tr,va,k[5:],device,ep,out,seed)
    if k=='fusion':
        from train_fusion import fit_fusion
        return fit_fusion(task,rec,tr,out,seed,device)
    if k=='stimulus_bag':
        from train_stimulus_bag import fit_bag
        return fit_bag(task,tr,va,out,device,ep,seed,threshold=rec['threshold'])
    if k=='population':
        from train_population import fit_population as method
    elif k=='rocket':
        from train_rocket import fit_rocket as method
    elif k=='kernel':
        from train_kernel import fit_kernel as method
    elif k=='neural':from train_neural import fit_neural as method
    elif k=='spectrum':from train_spectrum import fit_spectrum as method
    elif k=='tune':from train_tune import fit_tune as method
    elif k=='montage':from train_montage import fit_montage as method
    elif k=='full':from train_full import fit_full as method
    elif k=='motor_aligned':
        from motor_aligned import AlignedMotor
        from train_full import fit_full
        return fit_full(task,tr,va,out,device,ep,seed,model_class=AlignedMotor)
    elif k=='tree':from train_tree import fit_tree as method
    elif k=='geometry':from train_geometry import fit_geometry as method
    elif k=='motor':from train_motor import fit_motor as method
    elif k=='temporal':from train_temporal import fit_temporal as method
    elif k in ('steady','window','delay','source','evoked','canonical'):
        from train_special import fit_steady
        return fit_steady(task,tr,va,out,device,ep,seed,kind=k)
    else:raise ValueError(k)
    return method(task,tr,va,out,device,ep,seed)

def refit_once(task,rec,tr,out,seed,device,validation=None):
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'model.pt').is_file() and (out/'metrics.json').is_file():
            return
        return refit(task,rec,tr,out,seed,device,validation=validation)


def run(task):
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);torch.cuda.set_per_process_memory_fraction(.20)
    rec=json.loads((ROOT/'runs/selection.json').read_text())[task];kind=rec['architecture'];b=ROOT/'cache'/task
    groups=np.load(b/'groups.npy');y=np.load(b/'y.npy');raw=np.load(b/'x.npy',mmap_mode='r');mask=np.load(b/'mask.npy',mmap_mode='r')
    blind=np.isin(groups,json.loads((b/'splits.json').read_text())['blind']);tr=np.flatnonzero(~blind);va=np.flatnonzero(blind)
    info=public_info(task)
    for seed in SEEDS:
        out=ROOT/'runs/audit'/task/str(seed);p=out/'predictions.npz'
        if p.exists():continue
        refit_once(task,rec,tr,out,seed,'cuda')
        model=Predictor(ROOT,out/'model.pt',info,kind,'cuda');pred=[]
        with torch.inference_mode():
            for i in range(0,len(va),24):
                ids=va[i:i+24];z=model.run(torch.as_tensor(raw[ids],device='cuda'),torch.as_tensor(mask[ids],device='cuda')).cpu().numpy();pred.append(z)
        z=np.concatenate(pred);assert np.isfinite(z).all()
        np.savez_compressed(out/'pred.tmp.npz',idx=va,groups=groups[va],y=y[va],logits=z)
        (out/'pred.tmp.npz').replace(p)
        from sklearn.metrics import balanced_accuracy_score
        print(task,seed,'held-out BA',round(balanced_accuracy_score(y[va],z.argmax(1)),4),flush=True)
    final=ROOT/'runs/final'/task
    if not (final/'metrics.json').exists():refit(task,rec,np.arange(len(y)),final,SEEDS[0],'cuda')

if __name__=='__main__':run(sys.argv[1])
