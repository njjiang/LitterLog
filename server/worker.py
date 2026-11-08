"""Poll completed Protect events, classify clips, publish authenticated results."""
import argparse
import hashlib
import json
import getpass
import os
import sys
import time
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import requests
from pairing import pairing_token
from inference_clips import downloaded_clip
from clip_storage import StorageUnavailable
from thumbnail_store import publish_thumbnail
from backfill_thumbnails import ThumbnailRepair
from protect_service import ProtectService
from backend_paths import state_root, model_directory

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'inference'))

class ProtectAuthenticationError(Exception):
    pass

def authentication_failed(error):
    return error.response is not None and error.response.status_code in (401,403)

def check_protect_auth(response, **_kwargs):
    # The older byte downloader returns None for HTTP failures. Raise here too,
    # so its expired sessions follow the same recovery path as streamed exports.
    if response.status_code in (401,403):
        try:
            response.raise_for_status()
        finally:
            response.close()
    return response

class ProtectSession:
    def __init__(self, nvr, user, password):
        self.nvr=nvr
        self.user=user
        self.password=password
        self.retry_at=0
        self.retry_delay=30
        nvr.session.hooks.setdefault('response',[]).append(check_protect_auth)

    def defer_login(self):
        self.retry_at=time.monotonic()+self.retry_delay
        self.retry_delay=min(self.retry_delay*2,300)

    def login(self):
        self.nvr.session.cookies.clear()
        self.nvr.session.headers.pop('x-csrf-token',None)
        self.nvr.bootstrap=None
        try:
            # The dataset client prints the username after login.
            with redirect_stdout(None):
                self.nvr.login(self.user,self.password)
        except requests.HTTPError as error:
            if not authentication_failed(error):
                raise
            self.defer_login()
            raise ProtectAuthenticationError from None

    def call(self, operation, *args, **kwargs):
        if time.monotonic()<self.retry_at:
            raise ProtectAuthenticationError
        for attempt in range(2):
            try:
                result=operation(*args,**kwargs)
            except requests.HTTPError as error:
                if not authentication_failed(error):
                    raise
                if attempt:
                    self.defer_login()
                    raise ProtectAuthenticationError from None
                self.login()
                print('Protect session renewed.',flush=True)
            else:
                self.retry_at=0
                self.retry_delay=30
                return result

def scan_start(now, cursor, *, backfill_days=0):
    window=(backfill_days*86400 if backfill_days else 15*60)*1000
    start=min(now-window,cursor) if cursor is not None else now-window
    return start if backfill_days else max(start,now-7*86400*1000)

def report(stage):
    (state_root(ROOT)/'worker-status.json').write_text(json.dumps({'stage':stage,'time':datetime.now(timezone.utc).isoformat()}))
    print(stage,flush=True)

def configure_server_ca():
    """Scope the LitterLog CA to API requests, including older deployment settings.

    Requests lets REQUESTS_CA_BUNDLE override a Protect Session.verify value.
    Consume that legacy setting before creating Protect's independent session.
    """
    certificate = os.environ.get('LITTERLOG_SERVER_CA_CERT')
    if certificate is not None and not certificate:
        raise ValueError('LITTERLOG_SERVER_CA_CERT must identify a trusted CA file.')
    legacy = os.environ.pop('REQUESTS_CA_BUNDLE', None)
    return certificate or legacy or True

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,default=model_directory())
    parser.add_argument('--server',default='http://127.0.0.1:8767')
    parser.add_argument('--camera',required=True,help='Exact Protect camera name')
    parser.add_argument('--runtime',choices=['mlx','cpu'],default='mlx')
    parser.add_argument('--checkpoint',default='video_24f_balanced')
    parser.add_argument('--backfill-days',type=int,default=0)
    parser.add_argument('--backfill-only',action='store_true')
    args=parser.parse_args()
    if not 0 <= args.backfill_days <= 365: parser.error('Use 0–365 backfill days')
    server_ca=configure_server_ca()
    token=pairing_token()
    host=os.environ.get('LITTERLOG_NVR_HOST') or input('Protect host (hostname only): ')
    user=os.environ.get('LITTERLOG_NVR_USER') or input('Protect local username: ')
    password=os.environ.get('LITTERLOG_NVR_PASS') or getpass.getpass('Protect local password: ')
    report('Loading video model and checkpoint…')
    from download_events import UnifiProtect
    if args.runtime=='cpu':
        from cpu_infer import Classifier
        clf=Classifier(args.model_dir,checkpoint=args.checkpoint)
    else:
        import infer
        infer.LITTERLOG_DIR=args.model_dir
        infer.CKPT_ROOT=args.model_dir/'checkpoints'
        infer.LABELS_PATH=args.model_dir/'labels.txt'
        clf=infer.Classifier(ckpt_dir=args.checkpoint)
    trusted=state_root(ROOT)/'trusted-protect'/(hashlib.sha256(host.encode()).hexdigest()+'.pem')
    ca=os.environ.get('LITTERLOG_NVR_CA_CERT') or (str(trusted) if trusted.exists() else True)
    nvr=UnifiProtect(host,verify_ssl=ca)
    # Connect through the configured network address while verifying the explicit
    # identity in the already trusted certificate. Neither verification is disabled.
    tls_name_file=state_root(ROOT)/'protect-tls-name.txt'
    tls_name=os.environ.get('LITTERLOG_NVR_TLS_NAME') or (tls_name_file.read_text().strip() if tls_name_file.exists() else '')
    if tls_name:
        if ca is True:
            raise SystemExit('A trusted CA/certificate file is required for an explicit TLS identity.')
        from protect_tls import ProtectTLSAdapter
        nvr.session.mount(f'https://{host}/',ProtectTLSAdapter(tls_name))
    protect=ProtectSession(nvr,user,password)
    report('Connecting to Protect…')
    try:
        protect.login()
        cameras=protect.call(nvr.cameras)
    except ProtectAuthenticationError:
        raise SystemExit('Protect account authentication failed. Check the Protect local account credentials and permissions.') from None
    except requests.HTTPError:
        raise SystemExit('Protect connection failed. Check the Protect local account and service availability.') from None
    except requests.exceptions.ConnectionError as error:
        if not isinstance(error,requests.exceptions.SSLError):
            raise SystemExit('Could not reach Protect. Check LITTERLOG_NVR_HOST in ~/.zshrc and your local network connection.')
        raise SystemExit('Protect TLS verification failed. Check your saved certificate and TLS identity; verification remains enabled.')
    except requests.RequestException:
        raise SystemExit('Protect connection failed; check the Protect service and local network availability.') from None
    matches=[k for k,v in cameras.items() if v==args.camera]
    if len(matches)!=1: raise SystemExit('Camera name must identify exactly one camera.')
    camera=matches[0]
    # Scan a bounded recent window on restart; persistent event IDs deduplicate delivery.
    headers={'Authorization':'Bearer '+token}
    report('Connecting to LitterLog server…')
    try:
        known=requests.get(args.server.rstrip('/')+'/api/events/known',headers=headers,timeout=15,verify=server_ca)
        if known.status_code!=404:
            known.raise_for_status()
    except requests.HTTPError as error:
        if authentication_failed(error):
            raise SystemExit('LitterLog pairing token was rejected. Check LITTERLOG_APP_TOKEN on the worker and server.') from None
        raise SystemExit('LitterLog history request failed; check the server availability.') from None
    except requests.RequestException:
        raise SystemExit('Could not connect securely to the LitterLog server. Check its address and trusted certificate.') from None
    if known.status_code==404:
        print('Restart the robot server to enable history deduplication.',flush=True)
        seen=set()
    else:
        seen=set(known.json())
    first_scan=True
    scan_cursor=None
    db_path=Path(os.environ.get('LITTERLOG_DB',str(state_root(ROOT)/'events.sqlite3')))
    thumb_repair=ThumbnailRepair(db_path,state_root(ROOT)/'attention-cache',lambda:clf,ProtectService(),checkpoint=args.checkpoint,interval=30,limit=4)
    print('Watching completed Protect clips; no robot commands will be sent.',flush=True)
    while True:
        try:
            now=int(time.time()*1000)
            start_ms=scan_start(now,scan_cursor,backfill_days=args.backfill_days if first_scan else 0)
            # Keep the earliest uncompleted scan through authentication, storage,
            # and publishing outages. The next scan still overlaps recent events.
            scan_cursor=start_ms
            report('Checking Protect for completed events…')
            events=protect.call(nvr.events,start_ms,now,camera_ids=[camera])
            pending=[e for e in events if e.get('end') and e['id'] not in seen]
            print(f'{len(pending)} unprocessed completed clips in this scan.',flush=True)
            completed=0
            for event in sorted(pending,key=lambda e:e['start'],reverse=True):
                eid=event['id']
                if eid in seen or not event.get('end'): continue
                report('Downloading a completed camera clip…')
                start,end=event['start'],min(event['end'],event['start']+62000)
                with nullcontext(downloaded_clip(eid,camera,start,end,lambda:protect.call(nvr.download_clip,camera,start,end))) as clip:
                    report('Classifying camera clip…')
                    prediction=clf.predict(clip,max_seconds=60,with_attention=True)
                if prediction.best_frame:
                    try:
                        thumb_dir = state_root(ROOT) / 'attention-cache'
                        thumb_dir.mkdir(exist_ok=True)
                        publish_thumbnail(thumb_dir, eid, prediction, args.checkpoint)
                    except (OSError, ValueError):
                        raise
                response=requests.post(args.server.rstrip('/')+'/api/events',headers={'Authorization':'Bearer '+token},json={
                    'event_id':eid,'camera_id':camera,'start_ms':event['start'],'end_ms':event['end'],'time':datetime.fromtimestamp(event['start']/1000,timezone.utc).isoformat(),
                    'label':prediction.label,'confidence':prediction.confidence,
                    'cat':prediction.label if prediction.label in ('cat1','cat2') else ''},timeout=15,verify=server_ca)
                response.raise_for_status()
                seen.add(eid)
                completed+=1
                report(f'Backfill progress {completed}/{len(pending)} · '+prediction.label)
            if db_path.is_file():
                try:
                    thumb_repair.scan()
                except Exception as error:
                    print('Thumbnail repair check skipped ('+type(error).__name__+').',flush=True)
            # Bound memory; replaying older IDs is harmless because storage upserts.
            first_scan=False
            scan_cursor=now-15*60*1000
            if args.backfill_only:
                print('History import complete.',flush=True)
                return
        except StorageUnavailable:
            report('Clip storage unavailable; downloads will retry.')
        except ProtectAuthenticationError:
            print('Protect authentication failed. Retrying the Protect local account with a delay; check its credentials and permissions if this continues.',flush=True)
        except requests.HTTPError as error:
            if authentication_failed(error):
                print('LitterLog pairing token was rejected. Check LITTERLOG_APP_TOKEN on the worker and server.',flush=True)
            else: print('HTTP operation failed; retrying.',flush=True)
        except Exception as error:
            print('Clip processing failed ('+type(error).__name__+'); retrying.',flush=True)
        time.sleep(30)

if __name__=='__main__': main()
