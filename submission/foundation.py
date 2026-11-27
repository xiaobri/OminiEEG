"""Public CBraMod encoder and pooled intermediate representations."""
import torch
from torch import nn
import torch.nn.functional as F
from cbra.cbramod import CBraMod


class Foundation(nn.Module):
    def __init__(self,info,weights):
        super().__init__()
        c,t=info['input_shapes'][0]; self.fs=info['sampling_rate_hz']
        self.encoder=CBraMod()
        self.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.encoder.proj_out=nn.Identity()
        self.register_buffer('channels',torch.linspace(0,c-1,min(c,32)).round().long())
        self.pool_c=min(10,len(self.channels));self.pool_t=min(3,max(1,round(t/self.fs)))
    def patches(self,x,mask):
        x=x.float()[:,self.channels]*mask[:,self.channels,None]
        if self.fs!=200:
            t=max(200,round(x.shape[-1]*200/self.fs))
            x=F.adaptive_avg_pool1d(x,t)
        x=torch.nan_to_num(x,nan=0.,posinf=0.,neginf=0.).mul(0.01).clamp(-30,30)
        if x.shape[-1]%200: x=F.pad(x,(0,200-x.shape[-1]%200))
        return x.reshape(x.shape[0],x.shape[1],-1,200)
    def forward(self,x,mask):
        v=self.encoder.patch_embedding(self.patches(x,mask))
        vec=[]
        for i,layer in enumerate(self.encoder.encoder.layers):
            v=layer(v)
            if i in (3,11):
                pool=F.adaptive_avg_pool2d(v.permute(0,3,1,2).float(),(self.pool_c,self.pool_t))
                vec.append(pool.flatten(1))
        return torch.cat(vec,dim=1)

class MatrixPair:
    """Column-wise view of two row-aligned feature arrays."""
    def __init__(self,a,b):
        self.a=a;self.b=b
        assert len(a)==len(b)
        self.shape=(len(a),a.shape[1]+b.shape[1])
    def __getitem__(self,ids):
        import numpy as np
        return np.concatenate((self.a[ids],self.b[ids]),axis=-1).astype('float32',copy=False)
