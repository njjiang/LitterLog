"""CPU execution of the existing MLX checkpoint, with explicit tensor layout conversion."""
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
import av
import imageio_ffmpeg
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet18
from storage_paths import scratch_parent

def configure_cpu_threads():
    threads = int(os.environ.get('LITTERLOG_CPU_THREADS', '4'))
    if threads < 1:
        raise ValueError('LITTERLOG_CPU_THREADS must be a positive integer')
    torch.set_num_threads(threads)
    return threads

class Attention(nn.Module):
    def __init__(self,dim=128,heads=8):
        super().__init__()
        self.QKV=nn.Linear(dim,3*dim,bias=False)
        self.out=nn.Linear(dim,dim,bias=False)
        self.layernorm=nn.LayerNorm(dim,eps=1e-5)
        self.heads=heads
    def forward(self,x):
        b,t,dim=x.shape
        q,k,v=self.QKV(self.layernorm(x)).chunk(3,dim=-1)
        q,k,v=[a.reshape(b,t,self.heads,dim//self.heads) for a in (q,k,v)]
        scores=torch.softmax(torch.einsum('bqnd,bknd->bnqk',q,k)/math.sqrt(dim//self.heads),dim=-1)
        self.cls_attention=scores[0,:,0,1:].mean(dim=0).detach().cpu().numpy()
        values=torch.einsum('bnqk,bknd->bnqd',scores,v).permute(0,2,1,3).reshape(b,t,dim)
        return self.out(values)

class VideoModel(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.resnet=resnet18(weights=None)
        self.W=nn.Linear(512,128)
        self.attention=Attention()
        self.classifier=nn.Linear(128,classes)
        self.cls=nn.Parameter(torch.zeros(128))
        self.layernorm=nn.LayerNorm(128)
    def forward(self,x):
        b,t,h,w,c=x.shape
        frames=x.reshape(b*t,h,w,c).permute(0,3,1,2)
        features=frames
        for module in list(self.resnet.children())[:-1]: features=module(features)
        features=self.W(features.flatten(1)).reshape(b,t,128)
        sequence=torch.cat([self.cls.reshape(1,1,128).expand(b,1,128),features],dim=1)
        # Match model.py exactly: final LayerNorm is computed there but not used.
        pooled=self.attention(sequence)[:,0,:]
        return torch.softmax(self.classifier(pooled),dim=-1)

@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict
    best_frame: bytes | None = None
    best_frame_seconds: float | None = None
    attn_ratio: float | None = None

class Classifier:
    def __init__(self,model_dir,checkpoint='video_24f_balanced',n_frames=24):
        configure_cpu_threads()
        model_dir=Path(model_dir)
        self.labels=(model_dir/'labels.txt').read_text().split()
        self.n_frames=n_frames
        self.model=VideoModel(len(self.labels))
        with np.load(model_dir/'checkpoints'/checkpoint/'best.npz') as archive:
            converted={}
            for name in archive.files:
                key=name.replace('.layers.','.')
                values=archive[name]
                if values.ndim==4: values=values.transpose(0,3,1,2)
                converted[key]=torch.from_numpy(np.array(values,copy=True))
        missing,unexpected=self.model.load_state_dict(converted,strict=False)
        missing=[key for key in missing if not key.endswith('num_batches_tracked')]
        if missing or unexpected: raise ValueError(f'Checkpoint mapping failed: missing={missing}, unexpected={unexpected}')
        self.model.eval()
    def predict(self,video,max_seconds=60,with_attention=False):
        with av.open(str(video)) as container:
            duration=float(container.duration/av.time_base) if container.duration else 0
        with tempfile.TemporaryDirectory(prefix='litterlog-cpu-', dir=scratch_parent()) as tmp:
            args=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-hide_banner','-loglevel','error','-ss','1.0','-i',str(video)]
            if duration>2: args+=['-t',str(min(duration-2,max_seconds))]
            args+=['-vf','fps=2.0,scale=256:256','-q:v','4',str(Path(tmp)/'frame_%04d.jpg')]
            result=subprocess.run(args,capture_output=True)
            if result.returncode: raise RuntimeError('Frame extraction failed')
            paths=sorted(Path(tmp).glob('frame_*.jpg'))[:int(max_seconds*2)]
            if not paths: raise ValueError('No frames available')
            picked=[min(round(i*(len(paths)-1)/max(1,self.n_frames-1)),len(paths)-1) for i in range(self.n_frames)]
            frames=[]
            mean=np.array([.485,.456,.406],dtype=np.float32)
            std=np.array([.229,.224,.225],dtype=np.float32)
            for idx in picked:
                with Image.open(paths[idx]) as image:
                    image=image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240))
                    frames.append((np.asarray(image,dtype=np.float32)/255-mean)/std)
            with torch.inference_mode(): probs=self.model(torch.from_numpy(np.stack(frames)[None])).numpy()[0]
            best_frame=None; best_seconds=None; ratio=None
            if with_attention:
                attention=self.model.attention.cls_attention
                best_seconds=1.0+picked[int(attention.argmax())]/2.0
                ratio=float(attention.max()*self.n_frames)
                frame=Path(tmp)/'attention.jpg'
                result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-hide_banner','-loglevel','error','-ss',str(best_seconds),'-i',str(video),'-frames:v','1','-vf','scale=480:-2','-q:v','4',str(frame)],capture_output=True)
                if result.returncode==0 and frame.exists(): best_frame=frame.read_bytes()
        idx=int(probs.argmax())
        return Prediction(self.labels[idx],float(probs[idx]),dict(zip(self.labels,map(float,probs))),best_frame,best_seconds,ratio)
