"""Materialize public-training arrays in Lance row order. No fitted preprocessing."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'downstream/public-release/public_alias_20261118_001'
META = ROOT.parent / 'downstream/public_release_metadata/public_alias_20261118_001/public_train_window_metadata'


def prepare(task):
    info = json.loads((DATA/'task_manifest.json').read_text())['datasets'][task]
    c, t = info['input_shapes'][0]
    n = info['rows']
    output = ROOT/'cache'/task
    output.mkdir(parents=True, exist_ok=True)
    if (output/'info.json').exists():
        return
    ds = lance.dataset(DATA/task/'dataset.lance')
    assert ds.count_rows() == n
    values = np.lib.format.open_memmap(output/'x.npy', mode='w+', dtype='float32', shape=(n,c,t))
    mask = np.lib.format.open_memmap(output/'mask.npy', mode='w+', dtype='bool', shape=(n,c))
    y = np.zeros(n,dtype=np.int64)
    id_map = {}
    groups = np.zeros(n,dtype=np.int32)
    labels = {v:i for i,v in enumerate(info['label_values'])}
    row = 0
    seen_names = set()
    lengths = set()
    known = {}
    for p in sorted((META/task).glob('*.jsonl')):
        for line in p.open():
            a=json.loads(line)
            assert a['sample_id'] not in known
            known[a['sample_id']]=(a['label'],a['subject_id'])
    assert len(known)==n
    channels = None
    cols = ['data','shape','label','subject_id','sample_id','channel_names','channel_mask','valid_length']
    for b in ds.to_batches(columns=cols, batch_size=96):
        count = len(b)
        raw = b['data'].flatten().to_numpy(zero_copy_only=False).reshape(count,c,t)
        if b['channel_mask'].null_count:
            masks=np.asarray([v if v is not None else [True]*c for v in b['channel_mask'].to_pylist()],dtype=bool)
        else:
            masks=b['channel_mask'].flatten().to_numpy(zero_copy_only=False).reshape(count,c)
        names = b['channel_names'].to_pylist()
        if channels is None:
            channels=names[0]
        for names_i in set(map(tuple,names)):
            aliases={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
            if c>1 and [aliases.get(v,v) for v in names_i]!=[aliases.get(v,v) for v in channels]:
                raise ValueError(f'{task} has changing channel order')
            seen_names.add(names_i)
        assert all(shape==[c,t] for shape in b['shape'].to_pylist())
        if not np.isfinite(raw).all(): raw=np.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.)
        ids=b['sample_id'].to_pylist(); subs=b['subject_id'].to_pylist(); labs=b['label'].to_pylist()
        for i,(key,sub,lab) in enumerate(zip(ids,subs,labs)):
            assert known.pop(key)==(lab,sub)
            groups[row+i]=id_map.setdefault(sub,len(id_map))
            y[row+i]=labels[lab]
        values[row:row+count]=raw
        mask[row:row+count]=masks
        lengths.update(b['valid_length'].to_pylist())
        row+=count
    assert row==n and not known
    values.flush(); mask.flush()
    np.save(output/'groups.npy',groups); np.save(output/'y.npy',y)
    info.update(channel_names=channels,subject_ids=list(id_map),subjects=len(id_map),valid_lengths=sorted(lengths),
                counts=np.bincount(y,minlength=len(labels)).tolist())
    tmp=output/'info.tmp'; tmp.write_text(json.dumps(info,ensure_ascii=False,indent=2)); os.replace(tmp,output/'info.json')
    print(task,n,len(id_map),info['counts'],sorted(lengths)[:5],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    a=parser.parse_args()
    tasks=sorted(json.loads((DATA/'task_manifest.json').read_text())['datasets'],key=lambda t:(DATA/t/'dataset.lance/data').stat().st_size)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(prepare,tasks):pass
