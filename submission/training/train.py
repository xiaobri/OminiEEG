"""Reproduce a locked model on the public downstream release."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
PACKAGE = TRAINING.parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--downstream-dir', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--partition', choices=['all','development'], default='all')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = json.loads((TRAINING/'train_config.yaml').read_text())
    record = config['tasks'][args.task]
    seed = config['final_seed'] if args.seed is None else args.seed
    workspace = args.work_dir.resolve()
    if workspace == PACKAGE or PACKAGE in workspace.parents:
        parser.error('Choose a work directory outside the submission package.')
    if args.output.exists():
        parser.error('Choose a new checkpoint output path.')
    os.environ['OMNIEEG_TRAIN_WORKDIR'] = str(workspace)
    sys.path.insert(0,str(PACKAGE))
    import numpy as np
    import torch
    import prepare
    from features import make
    from finalize_models import refit
    from train_foundation import encoded

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        parser.error('Reproduction requires a CUDA GPU.')
    source = PACKAGE/'assets/cbramod-pretrained.pth'
    if digest(source) != config['backbone_sha256']:
        raise ValueError('Unexpected public CBraMod weights')
    asset = workspace/'assets'/source.name
    asset.parent.mkdir(parents=True,exist_ok=True)
    if not asset.exists():
        tmp = asset.with_name(f'{asset.name}.{os.getpid()}.tmp')
        shutil.copyfile(source,tmp)
        os.replace(tmp,asset)
    elif digest(asset) != config['backbone_sha256']:
        raise ValueError('Unexpected backbone in training workspace')
    release = config['release']
    prepare.ROOT = workspace
    prepare.DATA = args.downstream_dir.resolve()/'public-release'/release
    prepare.META = args.downstream_dir.resolve()/'public_release_metadata'/release/'public_train_window_metadata'
    prepare.prepare(args.task)
    base = workspace/'cache'/args.task
    make(args.task,'cuda')
    if record['architecture'] not in ('tune','full','montage','motor_aligned'):
        encoded(args.task,'cuda')
    groups = np.load(base/'groups.npy')
    if args.partition == 'development':
        split = json.loads((base/'splits.json').read_text())
        rows = np.flatnonzero(~np.isin(groups,split['blind']))
    else:
        rows = np.arange(len(groups))
    dest = workspace/'reproduction'/args.task/args.partition/str(seed)
    refit(args.task,record,rows,dest,seed,'cuda')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as dst, (dest/'model.pt').open('rb') as src:
        shutil.copyfileobj(src,dst)
    print(args.task,record['architecture'],len(rows),'training windows',args.output)


if __name__ == '__main__':
    main()
