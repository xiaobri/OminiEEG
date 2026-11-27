"""Retrain every declared task and assemble a fresh competition ZIP from the resulting weights."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downstream-dir',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',help='Comma-separated physical GPU IDs or device UUIDs. Defaults to the devices visible to PyTorch.')
    args=parser.parse_args()
    root=args.work_dir.resolve();output=args.output.resolve();downstream=args.downstream_dir.resolve()
    if PACKAGE in (root,*root.parents) or PACKAGE in (output,*output.parents) or output.exists():
        parser.error('Use an external work directory and a new output ZIP path.')
    manifest=json.loads((PACKAGE/'manifest.json').read_text())
    tasks={r['dataset_name']:r['checkpoint'] for r in manifest['tasks']}
    cfg=json.loads((PACKAGE/'training/train_config.yaml').read_text())
    if len(tasks)!=14 or set(tasks)!=set(cfg['tasks']) or len(set(tasks.values()))!=14:
        raise ValueError('Invalid task-to-checkpoint mapping')
    for name,digest in manifest['file_hashes'].items():
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('Unexpected package member')
        if name not in tasks.values() and sha(PACKAGE/name)!=digest:raise ValueError(f'Package file changed: {name}')
    if args.gpus: devices=args.gpus.split(',')
    elif os.environ.get('CUDA_VISIBLE_DEVICES'):devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    else:
        import torch
        devices=list(map(str,range(torch.cuda.device_count())))
    if not devices or any(not s.strip() for s in devices) or len(set(devices))!=len(devices):
        parser.error('Choose one or more distinct CUDA devices.')
    session=root/sha(PACKAGE/'manifest.json')
    session.mkdir(parents=True,exist_ok=True)
    with (session/'rebuild.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        identity=session/'source.json'
        binding={'downstream':str(downstream),'public_release':cfg['release']}
        if identity.exists() and json.loads(identity.read_text())!=binding:raise ValueError('Work directory belongs to different source data')
        identity.write_text(json.dumps(binding)+'\n')
        def worker(pair):
            gpu,assigned=pair
            for task in assigned:
                folder=session/task;folder.mkdir(exist_ok=True)
                weight=folder/'model.pt';receipt=folder/'weight.json'
                if weight.is_file() and receipt.is_file() and json.loads(receipt.read_text())['sha256']==sha(weight):
                    continue
                # Keep partial outputs out of the cache of validated checkpoints.
                with tempfile.TemporaryDirectory(dir=folder) as pending:
                    temp=Path(pending)/'model.pt'
                    command=[sys.executable,str(PACKAGE/'training/train.py'),'--task',task,'--downstream-dir',str(downstream),
                             '--work-dir',str(folder/'work'),'--output',str(temp)]
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONDONTWRITEBYTECODE':'1'}
                    with (folder/'train.log').open('a') as log:
                        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    digest=sha(temp)
                    os.replace(temp,weight)
                    new=folder/'receipt.pending'
                    new.write_text(json.dumps({'task':task,'sha256':digest,'bytes':weight.stat().st_size})+'\n')
                    new.replace(receipt)
                print(task,digest,flush=True)
        names=list(tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
            list(executor.map(worker,[(gpu,names[i::len(devices)]) for i,gpu in enumerate(devices)]))
        files={n:PACKAGE/n for n in manifest['file_hashes']}
        files.update({name:session/task/'model.pt' for task,name in tasks.items()})
        manifest['file_hashes']={name:sha(path) for name,path in sorted(files.items())}
        output.parent.mkdir(exist_ok=True,parents=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as temp:
            archive=Path(temp)/'new.zip'
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for name,path in sorted(files.items()):z.write(path,'submission/'+name)
                z.writestr('submission/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:raise ValueError('ZIP integrity failure')
                for name,digest in manifest['file_hashes'].items():
                    if hashlib.sha256(z.read('submission/'+name)).hexdigest()!=digest:raise ValueError(f'ZIP hash mismatch for {name}')
            # Atomic publish without replacing a previous output, including a concurrent run.
            os.link(archive,output)
        print(output,sha(output),flush=True)

if __name__=='__main__':main()
