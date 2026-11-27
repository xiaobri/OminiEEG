"""Portable evaluation of numerical LightGBM split trees, using NumPy only."""
import numpy as np

def encode_booster(booster):
    dump=booster.dump_model();k=int(dump.get('num_tree_per_iteration',1))
    trees=[]
    for item in dump['tree_info']:
        nodes=[]
        def encode(node):
            i=len(nodes);nodes.append(None)
            if 'leaf_value' in node:
                nodes[i]=(-1,0,0,float(node['leaf_value']))
            else:
                if node['decision_type']!='<=':raise ValueError('unexpected nonnumeric tree split')
                l=encode(node['left_child']);r=encode(node['right_child'])
                nodes[i]=(node['split_feature'],l,r,float(node['threshold']))
            return i
        encode(item['tree_structure']);trees.append(nodes)
    width=max(map(len,trees))
    feats=np.full((len(trees),width),-1,np.int32);lft=np.zeros_like(feats);right=lft.copy()
    vals=np.zeros(feats.shape,np.float64)
    for i,tree in enumerate(trees):
        for j,(a,b,c,d) in enumerate(tree):feats[i,j]=a;lft[i,j]=b;right[i,j]=c;vals[i,j]=d
    return dict(feats=feats,left=lft,right=right,val=vals,num_class=k)

def evaluate(tree,x):
    f,l,r,v=[tree[k] for k in ('feats','left','right','val')]
    kc=int(tree['num_class']);result=np.zeros((len(x),kc),dtype=np.float64)
    for start in range(0,len(f),96):
        ts=np.arange(start,min(start+96,len(f)))[None,:]
        ids=np.zeros((len(x),ts.size),dtype=np.int32)
        for _ in range(64):
            indices=f[ts,ids];active=indices>=0
            if not active.any():break
            feat=np.take_along_axis(x,np.maximum(indices,0),axis=1)
            ids=np.where(active,np.where(feat<=v[ts,ids],l[ts,ids],r[ts,ids]),ids)
        else:raise RuntimeError('tree traversal exceeded depth')
        value=v[ts,ids]
        for k in range(kc):result[:,k]+=value[:,(ts[0]%kc)==k].sum(axis=1)
    if kc==1:return np.column_stack([np.zeros(len(x)),result[:,0]]).astype('float32')
    return result.astype('float32')
