"""Protect camera discovery, live snapshots, and on-demand recording exports."""
import hashlib
import os
import sys
import threading
import time
import struct
import subprocess
import tempfile
import imageio_ffmpeg
from pathlib import Path
from urllib.parse import quote
import requests

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'inference'))
from download_events import UnifiProtect
from protect_tls import ProtectTLSAdapter
from backend_paths import state_root

def faststart_ready(path):
    """Check top-level MP4 boxes without reading the video payload."""
    with Path(path).open('rb') as source:
        return faststart_ready_source(source)


def faststart_ready_source(source):
    """Inspect an open, seekable recording."""
    length = source.seek(0, 2)
    source.seek(0)
    media_seen = False
    while source.tell() < length:
        start = source.tell()
        header = source.read(8)
        if len(header) != 8:
            return False
        size, kind = struct.unpack('>I4s', header)
        header_size = 8
        if size == 1:
            extended = source.read(8)
            if len(extended) != 8:
                return False
            size = struct.unpack('>Q', extended)[0]
            header_size = 16
        elif size == 0:
            size = length - start
        if size < header_size or start + size > length:
            return False
        if kind == b'moov':
            return not media_seen
        if kind in (b'mdat', b'moof'):
            media_seen = True
        source.seek(start + size)
    return False


def ensure_faststart(path):
    """Move MP4 metadata ahead of video, preserving encoded streams atomically."""
    path=Path(path)
    if faststart_ready(path): return
    fd,name=tempfile.mkstemp(prefix='.litterlog-faststart-',suffix='.mp4',dir=path.parent)
    os.close(fd)
    optimized=Path(name)
    try:
        # Stream copy avoids the cost and quality loss of video re-encoding.
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
                        '-nostdin','-y','-i',str(path),'-map','0:v:0','-map','0:a?',
                        '-c','copy','-movflags','+faststart',str(optimized)],
                       check=True,timeout=120,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if not faststart_ready(optimized): raise ValueError('Could not prepare recording for streaming')
        optimized.replace(path)
    finally:
        optimized.unlink(missing_ok=True)


class ProtectService:
    def __init__(self):
        self.lock=threading.RLock()
        self.client=None
        self.camera=None
    def connect(self):
        if self.client is not None: return
        host=os.environ.get('LITTERLOG_NVR_HOST','')
        user=os.environ.get('LITTERLOG_NVR_USER','')
        password=os.environ.get('LITTERLOG_NVR_PASS','')
        if not all((host,user,password)): raise ValueError('Protect credentials are not configured')
        trusted=state_root(ROOT)/'trusted-protect'/(hashlib.sha256(host.encode()).hexdigest()+'.pem')
        ca=os.environ.get('LITTERLOG_NVR_CA_CERT') or (str(trusted) if trusted.exists() else True)
        client=UnifiProtect(host,verify_ssl=ca)
        identity=state_root(ROOT)/'protect-tls-name.txt'
        name=os.environ.get('LITTERLOG_NVR_TLS_NAME') or (identity.read_text().strip() if identity.exists() else '')
        if name and ca is not True: client.session.mount(f'https://{host}/',ProtectTLSAdapter(name))
        # Keep login credentials and user identifiers out of service logs.
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()): client.login(user,password)
        cameras=client.get_bootstrap().get('cameras',[])
        wanted=os.environ.get('LITTERLOG_CAMERA_NAME','Litter Box')
        matches=[c for c in cameras if c.get('name')==wanted or c.get('id')==wanted]
        if len(matches)!=1: raise ValueError('Set LITTERLOG_CAMERA_NAME to exactly one Protect camera')
        self.camera=matches[0]
        self.client=client
    def get(self,path,**kwargs):
        self.connect()
        response=self.client.session.get(self.client.base+path,timeout=120,**kwargs)
        if response.status_code in (401,403):
            response.close(); self.reconnect()
            response=self.client.session.get(self.client.base+path,timeout=120,**kwargs)
        response.raise_for_status()
        return response
    def reconnect(self):
        previous=self.client
        self.client=None
        if previous is not None:
            previous.session.close()
        self.connect()
    def events(self,start_ms,end_ms):
        self.connect()
        try:
            return self.client.events(start_ms,end_ms,camera_ids=[self.camera['id']])
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code not in (401,403):
                raise
            error.response.close()
            self.reconnect()
            return self.client.events(start_ms,end_ms,camera_ids=[self.camera['id']])
    def snapshot(self):
        with self.lock:
            self.connect()
            response=self.get('/proxy/protect/api/cameras/'+quote(self.camera['id'],safe='')+'/snapshot',params={'ts':int(time.time()*1000),'force':'true'})
            if not response.headers.get('Content-Type','').startswith('image/'): raise ValueError('Protect did not return an image')
            return response.content
    def recent(self,days=7):
        with self.lock:
            self.connect()
            now=int(time.time()*1000)
            records=self.events(now-int(days*86400000),now)
            return self.completed_records(records,now)
    def completed_records(self,events,now_ms):
        """Accept only completed intervals from the configured camera."""
        records=[]
        for event in events:
            camera=event.get('camera')
            if isinstance(camera,dict): camera=camera.get('id')
            start,end=event.get('start'),event.get('end')
            if (camera!=self.camera['id'] or not isinstance(event.get('id'),str) or not event['id']
                    or type(start) is not int or type(end) is not int or not 0<start<end<=now_ms):
                continue
            records.append(dict(event_id=event['id'],camera_id=camera,start_ms=start,end_ms=end))
        return records
    def recordings_for(self,event_ids,start_ms,end_ms):
        """Resolve legacy classifier events by exact Protect ID, never by nearest time."""
        with self.lock:
            self.connect()
            events=self.events(start_ms,end_ms)
            return [r for r in self.completed_records(events,end_ms) if r['event_id'] in event_ids]
    def export(self,event_id,camera_id,start_ms,end_ms,destination,*,max_bytes=1024**3):
        # Protect session/login access is serialized, but its response body can be
        # streamed without holding up snapshots or another recording request.
        with self.lock:
            self.connect()
            if camera_id!=self.camera['id']: raise ValueError('Recording is not from the configured camera')
            response=self.get('/proxy/protect/api/video/export',params={'camera':camera_id,'start':start_ms,'end':end_ms},stream=True)
        try:
            if 'video' not in response.headers.get('Content-Type','') and 'octet-stream' not in response.headers.get('Content-Type',''):
                raise ValueError('Recording unavailable or outside retention')
            destination=Path(destination)
            fd,name=tempfile.mkstemp(prefix='.litterlog-export-',suffix='.partial',dir=destination.parent)
            partial=Path(name)
            total=0
            try:
                with os.fdopen(fd,'wb') as f:
                    for chunk in response.iter_content(1024*1024):
                        total+=len(chunk)
                        if total>max_bytes:
                            raise ValueError(f'Recording exceeds {max_bytes // 1024**2} MiB export limit')
                        f.write(chunk)
                if total==0: raise ValueError('Empty recording')
                ensure_faststart(partial)
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        finally: response.close()
