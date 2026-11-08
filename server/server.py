"""Authenticated iPhone backend with persistent Whisker session and event storage."""
import asyncio
from contextlib import closing, contextmanager, nullcontext, suppress
import getpass
import hashlib
import json
import math
import os
import secrets
import sqlite3
import ssl
import stat
import sys
import tempfile
import re
import imageio_ffmpeg
from urllib.parse import quote
from datetime import datetime, timezone
from pathlib import Path
from aiohttp import web
from pairing import pairing_token
from protect_service import ProtectService, faststart_ready, ensure_faststart
from attribution import identify, match_weight
from backend_paths import state_root
import clip_storage
from thumbnail_store import thumbnail_metadata
from pylitterbot import Account, LitterRobot4
from pylitterbot.enums import LitterBoxStatus

ROOT = Path(__file__).resolve().parent
DB = Path(os.environ.get('LITTERLOG_DB', str(state_root(ROOT) / 'events.sqlite3')))
RECORDING_SYNC_INTERVAL = 30
COMMAND_TIMEOUT = 40  # The iPhone gives a clean request 45 seconds.
RESET_READY_TIMEOUT = 12
RESET_POLL_INTERVAL = 1
CLIP_STREAM_CHUNK = 256 * 1024
HISTORY_ARCHIVE_INTERVAL = 24 * 60 * 60
HISTORY_ARCHIVE_RETRY = 5 * 60
THUMBNAIL_INDEX_INTERVAL = 20

TABLES = {
    'events': ('id INTEGER PRIMARY KEY, event_id TEXT UNIQUE, time TEXT, label TEXT, '
               'confidence REAL, cat TEXT'),
    'recordings': 'event_id TEXT PRIMARY KEY, camera_id TEXT, start_ms INTEGER, end_ms INTEGER',
    'overrides': 'event_id INTEGER PRIMARY KEY, label TEXT, cat TEXT',
    'weights': 'id TEXT PRIMARY KEY, time TEXT, weight_lbs REAL',
}
NEXT_EVENT_ID_SQL = (
    '(MAX(COALESCE((SELECT MAX(id) FROM events), 0), '
    'COALESCE((SELECT event_id_floor FROM history_archive_state WHERE singleton = 1), 0)) + 1)'
)

def ensure_archive_schema(connection):
    connection.execute('CREATE TABLE IF NOT EXISTS history_archive_state ('
                       'singleton INTEGER PRIMARY KEY CHECK (singleton = 1), '
                       'event_id_floor INTEGER NOT NULL)')

def store():
    c = sqlite3.connect(DB,uri=True)
    for table,schema in TABLES.items():
        c.execute(f'CREATE TABLE IF NOT EXISTS {table} ({schema})')
    ensure_archive_schema(c)
    return c


@contextmanager
def history_store(archive):
    # The archive is a read-only tmpfs replica. Hold the same lock used for
    # swapping it and pruning SQLite so a query sees both sides of the move.
    with archive.lock if archive is not None else nullcontext():
        with closing(store()) as c, c:
            remote = archive is not None and archive.path.is_file()
            if remote:
                c.execute('ATTACH DATABASE ? AS archived',(archive.path.as_uri()+'?mode=ro&immutable=1',))
            for table,key in (('events','id'),('recordings','event_id'),('weights','id'),('overrides','event_id')):
                sql=f'CREATE TEMP VIEW history_{table} AS SELECT * FROM main.{table}'
                if remote:
                    sql+=f' UNION ALL SELECT a.* FROM archived.{table} a WHERE NOT EXISTS (SELECT 1 FROM main.{table} l WHERE l.{key}=a.{key})'
                c.execute(sql)
            yield c


def require_archived_history(c, archive):
    if (archive is not None and not archive.path.is_file() and
            c.execute('SELECT 1 FROM history_archive_state WHERE singleton=1').fetchone()):
        raise web.HTTPServiceUnavailable(text='Archived history is temporarily unavailable. Try again when the NAS reconnects.')


async def history_archive_lifecycle(app):
    archive=app.get('history_archive')
    if archive is None:
        yield
        return
    with closing(store()):
        pass

    async def monitor():
        while True:
            job=asyncio.create_task(asyncio.to_thread(archive.run,months=app['history_retention_months']))
            try:
                counts=await asyncio.shield(job)
            except asyncio.CancelledError:
                with suppress(Exception):
                    await job
                raise
            except Exception as error:
                print(f'LitterLog history archive will retry ({type(error).__name__}).',flush=True)
                await asyncio.sleep(HISTORY_ARCHIVE_RETRY)
            else:
                print('LitterLog history archive check complete. Moved '+json.dumps(counts,sort_keys=True)+'.',flush=True)
                await asyncio.sleep(HISTORY_ARCHIVE_INTERVAL)

    task=asyncio.create_task(monitor())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    archive.close()

def normalize(data):
    label = data.get('label')
    confidence = float(data.get('confidence', -1))
    cat = data.get('cat', label if label in ('cat1', 'cat2') else '')
    event_id = data.get('event_id')
    timestamp = data.get('time', datetime.now(timezone.utc).isoformat())
    if not isinstance(label, str) or not 1 <= len(label) <= 80 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid label/confidence')
    if not isinstance(cat, str) or len(cat) > 80 or (event_id is not None and (not isinstance(event_id, str) or len(event_id)>200)):
        raise ValueError('Invalid cat/event ID')
    if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp.replace('Z','+00:00')).tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return event_id, timestamp, label, confidence, cat

@web.middleware
async def auth(request, handler):
    expected = request.app['token']
    credential = request.headers.get('Authorization','')
    media_query = request.method == 'GET' or (request.method == 'HEAD' and 'token' in request.query)
    if media_query and request.path.startswith('/media/'):
        credential = 'Bearer ' + request.query.get('token','')
    if not secrets.compare_digest(credential, 'Bearer ' + expected):
        raise web.HTTPUnauthorized(text='Authentication required')
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except clip_storage.StorageUnavailable:
        return web.json_response({'error':'Clip storage is unavailable. Check clip storage and try again.'},status=503)
    except (ValueError, TypeError, KeyError):
        return web.json_response({'error':'Invalid request'}, status=400)
    except Exception:
        return web.json_response({'error':'Service unavailable; no successful command confirmed'}, status=502)

async def lifecycle(app):
    app['account'] = None
    app['robot'] = None
    app['lock'] = asyncio.Lock()
    async def monitor():
        while True:
            try:
                async with app['lock']:
                    if app['account'] is None:
                        account = Account()
                        try:
                            await asyncio.wait_for(account.connect(username=app['email'], password=app['password'], load_robots=True),45)
                            candidates = [r for r in account.robots if isinstance(r,LitterRobot4)]
                            selected = os.environ.get('LITTERLOG_ROBOT_ID')
                            if selected: candidates = [r for r in candidates if r.id == selected]
                            if len(candidates) != 1:
                                print('Select exactly one LR4 using LITTERLOG_ROBOT_ID. Available: '+json.dumps([{'id':r.id,'name':r.name} for r in account.robots if isinstance(r,LitterRobot4)]), flush=True)
                                raise ValueError('Robot selection required')
                            app['account'], app['robot'] = account, candidates[0]
                        except BaseException:
                            await account.disconnect()
                            raise
                    await asyncio.wait_for(app['robot'].refresh(),20)
                    app['updated'] = datetime.now(timezone.utc).isoformat()
                    if asyncio.get_running_loop().time()-app.get('weights_fetched',-1000)>300:
                        try:
                            readings=await asyncio.wait_for(app['robot'].get_activity_history(limit=1000),30)
                            with history_store(app.get('history_archive')) as c:
                                require_archived_history(c,app.get('history_archive'))
                                for entry in readings:
                                    found=re.fullmatch(r'Pet Weight Recorded: ([0-9.]+) lbs',str(entry.action))
                                    if found:
                                        weight=float(found.group(1))
                                        if not math.isfinite(weight) or weight<=0: continue
                                        timestamp=entry.timestamp.isoformat()
                                        reading_id=timestamp+':'+str(weight)
                                        c.execute('INSERT INTO weights SELECT ?,?,? WHERE NOT EXISTS (SELECT 1 FROM history_weights WHERE id=?)',
                                                  (reading_id,timestamp,weight,reading_id))
                            app['weights_fetched']=asyncio.get_running_loop().time()
                        except Exception:
                            print('Weight history unavailable; robot status remains connected.',flush=True)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                print('Whisker connection unavailable ('+type(error).__name__+'); retrying.',flush=True)
                async with app['lock']:
                    if app['account']:
                        await app['account'].disconnect()
                    app['account'] = app['robot'] = None
            await asyncio.sleep(30)
    task = asyncio.create_task(monitor())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass
    if app['account']: await app['account'].disconnect()

RESETTABLE_STATUSES = (
    LitterBoxStatus.CAT_SENSOR_TIMING,
    LitterBoxStatus.CAT_DETECTED,
    LitterBoxStatus.CAT_SENSOR_INTERRUPTED,
    LitterBoxStatus.CAT_SENSOR_FAULT,
    LitterBoxStatus.PAUSED,
)

async def status(request):
    app=request.app
    async with app['lock']:
        r=app['robot']
        if r is None: return web.json_response({'connected':False,'state':'Connecting to Whisker','supported_commands':[],'last_command_result':app.get('last_command_result')})
        can_clean=(r.is_online and r.status in (LitterBoxStatus.READY,*RESETTABLE_STATUSES)
                   and asyncio.get_running_loop().time()-app.get('last_command',-1000)>=60)
        supported=(['cycle'] if can_clean else [])+(['pause','reset'] if r.is_online else [])
        return web.json_response({'connected':r.is_online,'name':r.name,'state':r.status_text or str(r.status.value),
            'waste_percent':r.waste_drawer_level,'litter_level':f'{r.litter_level:.0f}%',
            'supported_commands':supported,'updated_at':app.get('updated'),'last_weight_lbs':r.pet_weight,'cycle_count':r.cycle_count,'wait_time_minutes':r.clean_cycle_wait_time_minutes,'hopper_status':r.hopper_status.value if r.hopper_status else None,'hopper_enabled':not r.is_hopper_removed if r.hopper_status else None,
            'last_command_result':app.get('last_command_result')})


def clean_response(app,payload,status=200,message=None):
    # Record only our controlled messages, never credentials, cloud responses,
    # request URLs, or device identifiers. Exposed through authenticated status.
    if message is None:
        message=payload.get('error') or ('Robot is already cleaning.' if payload.get('already_cleaning') else 'Clean request accepted.')
    outcome={'time':datetime.now(timezone.utc).isoformat(),'status_code':status,'message':message}
    app['last_command_result']=outcome
    print('Robot command result: '+json.dumps(outcome),flush=True)
    return web.json_response(payload,status=status)

async def command(request):
    data=await request.json()
    cmd=data.get('command')
    if cmd not in ('cycle','reset','pause'):
        return web.json_response({'error':'Unsupported command'},status=422)
    app=request.app
    label={'cycle':'clean','reset':'reset','pause':'pause'}[cmd]
    try:
        # Bound lock acquisition, reset acknowledgement, status polling, and
        # dispatch together so a late command is not sent after the phone times out.
        async with asyncio.timeout(COMMAND_TIMEOUT):
            if cmd == 'cycle':
                return await request_clean_cycle(app)
            if cmd == 'reset':
                return await request_reset(app)
            return await request_pause(app)
    except TimeoutError:
        return clean_response(app,{'error':f'The {label} request timed out. Check the robot status before retrying.'},status=504)
    except Exception:
        return clean_response(app,{'error':f'Could not confirm the {label} request. Check the robot status before retrying.'},status=502)


async def request_reset(app):
    async with app['lock']:
        r=app['robot']
        if r is None:
            return clean_response(app,{'error':'Whisker is not connected yet. Try again once the robot reconnects.'},status=503)
        await asyncio.wait_for(r.refresh(),8)
        if not r.is_online:
            return clean_response(app,{'error':'The robot is offline.'},status=409)
        accepted=await asyncio.wait_for(r.reset(),10)
        if accepted is not True:
            return clean_response(app,{'error':'Whisker did not confirm the reset request.'},status=502)
        now=asyncio.get_running_loop().time()
        app['last_reset']=now
        app['last_command']=-1000
        with suppress(Exception):
            await asyncio.wait_for(r.refresh(),5)
            app['updated']=datetime.now(timezone.utc).isoformat()
        return clean_response(app,{'accepted':True},message='Reset request accepted.')


async def request_pause(app):
    async with app['lock']:
        r=app['robot']
        if r is None:
            return clean_response(app,{'error':'Whisker is not connected yet. Try again once the robot reconnects.'},status=503)
        await asyncio.wait_for(r.refresh(),8)
        if not r.is_online:
            return clean_response(app,{'error':'The robot is offline.'},status=409)
        if r.status == LitterBoxStatus.PAUSED:
            return clean_response(app,{'accepted':True},message='Robot is already paused.')
        accepted=await asyncio.wait_for(r.reset(),10)
        if accepted is not True:
            return clean_response(app,{'error':'Whisker did not confirm the pause request.'},status=502)
        with suppress(Exception):
            await asyncio.wait_for(r.refresh(),5)
            app['updated']=datetime.now(timezone.utc).isoformat()
        return clean_response(app,{'accepted':True},message='Pause request accepted.')


async def request_clean_cycle(app):
    async with app['lock']:
        r=app['robot']
        if r is None: return clean_response(app,{'error':'Whisker is not connected yet. Try again once the robot reconnects.'},status=503)
        await asyncio.wait_for(r.refresh(),8)
        if r.is_online and r.status == LitterBoxStatus.CLEAN_CYCLE:
            # The requested state already exists; acknowledge without sending
            # another command, even if the phone showed an older Ready state.
            return clean_response(app,{'accepted':True,'already_cleaning':True})
        if not r.is_online or r.status not in (LitterBoxStatus.READY,*RESETTABLE_STATUSES):
            return clean_response(app,{'error':'Cleaning is available only when the robot is Ready or its cycle timer is counting down.'},status=409)
        now=asyncio.get_running_loop().time()
        if now - app.get('last_command', -1000) < 60:
            return clean_response(app,{'error':'A clean request was recently submitted'},status=429)
        # Reserve before the FIRST mutation, including reset. An uncertain
        # acknowledgement may mean Whisker already acted; never auto-retry it.
        app['last_command']=now
        if r.status in RESETTABLE_STATUSES:
            recent_reset=(now - app.get('last_reset', -1000)) < 10
            if not recent_reset:
                accepted=await asyncio.wait_for(r.reset(),10)
                if accepted is not True:
                    return clean_response(app,{'error':'Whisker did not confirm the timer reset. No clean command was sent.'},status=502)
                app['last_reset']=asyncio.get_running_loop().time()
            try:
                async with asyncio.timeout(RESET_READY_TIMEOUT):
                    polls=0
                    while True:
                        await asyncio.wait_for(r.refresh(),5)
                        if not r.is_online:
                            return clean_response(app,{'error':'The robot went offline after the timer reset. No clean command was sent.'},status=409)
                        if r.status == LitterBoxStatus.CLEAN_CYCLE:
                            # Reset can itself trigger a cycle. Do not send a
                            # second clean request when fresh status confirms it.
                            return clean_response(app,{'accepted':True,'already_cleaning':True})
                        if r.status == LitterBoxStatus.READY:
                            break
                        if r.status not in RESETTABLE_STATUSES:
                            return clean_response(app,{'error':'The robot is not Ready after the timer reset. No clean command was sent.'},status=409)
                        polls+=1
                        if recent_reset and polls == 3:
                            accepted=await asyncio.wait_for(r.reset(),10)
                            if accepted is not True:
                                return clean_response(app,{'error':'Whisker did not confirm the timer reset. No clean command was sent.'},status=502)
                        await asyncio.sleep(RESET_POLL_INTERVAL)
            except TimeoutError:
                return clean_response(app,{'error':'The timer reset was sent, but Ready was not confirmed. No clean command was sent; check the robot status.'},status=504)
        accepted=await asyncio.wait_for(r.start_cleaning(),10)
        if accepted is not True: return clean_response(app,{'error':'Whisker did not confirm acceptance'},status=502)
        with suppress(Exception):
            await asyncio.wait_for(r.refresh(),5)
            app['updated']=datetime.now(timezone.utc).isoformat()
        return clean_response(app,{'accepted':True})

async def events(request):
    archive=request.app.get('history_archive')
    if request.method=='GET':
        offset=max(0,int(request.query.get('offset','0')))
        rows=attributed_events(archive=archive,**thumbnail_index(request.app))
        rows.sort(key=lambda e:(e['time'],e['id']),reverse=True)
        return web.json_response(rows[offset:offset+100])
    data=await request.json()
    values=normalize(data)
    with history_store(archive) as c:
        previous=c.execute('SELECT id FROM history_events WHERE event_id=?',(values[0],)).fetchone()
        if (previous is None and archive is not None and not archive.path.is_file()
                and archive.expired(values[1],months=request.app['history_retention_months'])):
            require_archived_history(c,archive)
        c.execute(f'INSERT INTO events(id,event_id,time,label,confidence,cat) VALUES(COALESCE(?,{NEXT_EVENT_ID_SQL}),?,?,?,?,?) ON CONFLICT(event_id) DO UPDATE SET time=excluded.time,label=excluded.label,confidence=excluded.confidence,cat=excluded.cat',
                  (previous[0] if previous else None,*values))
        if data.get('camera_id') and data.get('start_ms') and data.get('end_ms'):
            start,end=int(data['start_ms']),int(data['end_ms'])
            if end<=start: raise ValueError('Invalid recording interval')
            c.execute('INSERT OR REPLACE INTO recordings VALUES(?,?,?,?)',(data['event_id'],str(data['camera_id']),start,end))
    return web.json_response({'ok':True},status=201)

def thumbnail_index(app):
    return dict(thumbnail_names=app.get('thumbnail_index',{}).get('names',()))


def attributed_events(*,archive=None,thumbnail_names=()):
    with history_store(archive) as c:
        c.row_factory=sqlite3.Row
        rows=c.execute('SELECT e.id,e.time,e.label,e.confidence,e.cat,e.event_id,r.camera_id,r.start_ms,r.end_ms,o.label AS manual_label,o.cat AS manual_cat FROM history_events e LEFT JOIN history_recordings r ON e.event_id=r.event_id LEFT JOIN history_overrides o ON e.id=o.event_id').fetchall()
    values=[]
    thumbnails = state_root(ROOT) / 'attention-cache'
    for row in rows:
        e=dict(row)
        e['original_label']=e['label']; e['original_cat']=e['cat']
        if e['manual_label'] is not None: e['label']=e['manual_label']
        e['has_video']=bool(e.pop('camera_id'))
        name=hashlib.sha256((e.get('event_id') or '').encode()).hexdigest()
        metadata = thumbnail_metadata(thumbnails, name) if name not in thumbnail_names else None
        available = (name in thumbnail_names or (thumbnails / (name + '.jpg')).exists())
        e['attention_thumbnail'] = '/media/thumbs/' + name + '.jpg' if available else None
        e['attention_seconds'] = metadata.get('seconds') if metadata else None
        values.append(e)
    return identify(values)

async def override_prediction(request):
    event_id=int(request.match_info['id'])
    data=await request.json()
    archive=request.app.get('history_archive')
    with history_store(archive) as c:
        if not c.execute('SELECT 1 FROM history_events WHERE id=?',(event_id,)).fetchone():
            require_archived_history(c,archive)
            raise web.HTTPNotFound()
        if data.get('reset') is True:
            require_archived_history(c,archive)
            if archive is not None:
                # An upload might already have captured the previous correction.
                c.execute('INSERT OR REPLACE INTO overrides VALUES(?,NULL,NULL)',(event_id,))
            else:
                c.execute('DELETE FROM overrides WHERE event_id=?',(event_id,))
        else:
            cat=data.get('cat'); label=data.get('label')
            if cat not in ('','cat1','cat2') or label not in ('recording','cat1','cat2','none','pee','poo','both'): raise ValueError('Invalid correction')
            c.execute('INSERT OR REPLACE INTO overrides VALUES(?,?,?)',(event_id,label,cat))
    return web.json_response({'ok':True})

async def cat_activity(request):
    from cat_activity import visits
    return web.json_response(visits(attributed_events(archive=request.app.get('history_archive'),**thumbnail_index(request.app))))

async def weights(request):
    archive=request.app.get('history_archive')
    with history_store(archive) as c:
        c.row_factory=sqlite3.Row
        readings=[dict(r) for r in c.execute('SELECT * FROM history_weights ORDER BY time DESC')]
    events=attributed_events(archive=archive,**thumbnail_index(request.app))
    event_by_id={event['id']:event for event in events}
    results=[]
    for reading in readings:
        matched=match_weight(reading,events)
        # Send the exact evidence with the reading so older weigh-ins do not
        # depend on which pages of Activity the phone has already loaded.
        results.append(dict(matched,weight_kg=reading['weight_lbs']*.45359237,
                            identity_event=event_by_id.get(matched['identity_event_id']),
                            video_event=event_by_id.get(matched['video_event_id'])))
    return web.json_response(results)

async def known_events(request):
    with history_store(request.app.get('history_archive')) as c:
        rows=c.execute("SELECT event_id FROM history_events WHERE event_id IS NOT NULL AND label != 'recording'").fetchall()
    return web.json_response([row[0] for row in rows])

async def media_lifecycle(app):
    source=os.environ.get('LITTERLOG_RTSP_URL')
    app['media_dir']=None
    if not source:
        yield
        return
    try:
        folder=tempfile.TemporaryDirectory(prefix='litterlog-hls-',dir=clip_storage.scratch_parent())
    except clip_storage.StorageUnavailable:
        print('Live relay storage is unavailable. Check clip storage and restart the server.',flush=True)
        yield
        return
    app['media_dir']=Path(folder.name)
    process=None
    try:
        process=await asyncio.create_subprocess_exec(imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-rtsp_transport','tcp','-i',source,'-an','-c:v','libx264','-preset','veryfast','-tune','zerolatency','-pix_fmt','yuv420p','-vf','scale=1280:-2','-g','48','-sc_threshold','0','-f','hls','-hls_time','2','-hls_list_size','6','-hls_flags','delete_segments+temp_file','-hls_segment_filename',str(Path(folder.name)/'part%09d.ts'),str(Path(folder.name)/'live.m3u8'),stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
        yield
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try: await asyncio.wait_for(process.wait(),5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        folder.cleanup()

async def camera(request):
    directory=request.app.get('media_dir')
    if directory is not None and (directory/'live.m3u8').exists():
        return web.json_response({'mode':'video','url':'/media/live.m3u8?token='+quote(request.app['token'],safe='')})
    # A trusted authenticated snapshot feed works without enabling RTSP on the NVR.
    await asyncio.to_thread(request.app['protect'].snapshot)
    return web.json_response({'mode':'snapshots','url':'/media/snapshot.jpg?token='+quote(request.app['token'],safe='')})

async def live_snapshot(request):
    try: data=await asyncio.to_thread(request.app['protect'].snapshot)
    except Exception: return web.json_response({'error':'Protect camera unavailable. Check credentials, camera name and certificate.'},status=503)
    return web.Response(body=data,content_type='image/jpeg',headers={'Cache-Control':'no-store'})

async def import_recordings(request):
    data=await request.json()
    days=int(data.get('days',7))
    if not 1<=days<=90: raise ValueError('Use 1–90 days')
    records=await asyncio.to_thread(request.app['protect'].recent,days)
    archive=request.app.get('history_archive')
    with history_store(archive) as c:
        require_archived_history(c,archive)
        for r in records:
            c.execute('INSERT OR REPLACE INTO recordings VALUES(?,?,?,?)',(r['event_id'],r['camera_id'],r['start_ms'],r['end_ms']))
            c.execute(f'INSERT INTO events(id,event_id,time,label,confidence,cat) SELECT {NEXT_EVENT_ID_SQL},?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM history_events WHERE event_id=?)',
                      (r['event_id'],datetime.fromtimestamp(r['start_ms']/1000,timezone.utc).isoformat(),'recording',0,'',r['event_id']))
    return web.json_response({'imported':len(records)})


async def reconcile_recordings(app):
    """Repair links written by older workers without changing their predictions."""
    now_ms=int(datetime.now(timezone.utc).timestamp()*1000)
    cutoff=now_ms-7*86400000
    with store() as c:
        rows=c.execute('SELECT e.event_id,e.time FROM events e LEFT JOIN recordings r '
                       'ON e.event_id=r.event_id WHERE r.event_id IS NULL '
                       'AND e.event_id IS NOT NULL AND e.event_id != ?',('',)).fetchall()
    pending={}
    for event_id,timestamp in rows:
        try:
            date=datetime.fromisoformat(timestamp.replace('Z','+00:00'))
            if date.tzinfo is None: continue
            start_ms=int(date.timestamp()*1000)
            if cutoff<=start_ms<=now_ms: pending[event_id]=start_ms
        except (ValueError,TypeError,AttributeError,OverflowError):
            continue
    if not pending: return 0
    records=await asyncio.to_thread(app['protect'].recordings_for,set(pending),
                                   max(cutoff,min(pending.values())-60000),now_ms)
    repaired=0
    with store() as c:
        for record in records:
            if record['event_id'] not in pending: continue
            # A worker/import may have supplied newer metadata during the fetch.
            cursor=c.execute('INSERT OR IGNORE INTO recordings VALUES(?,?,?,?)',
                             (record['event_id'],record['camera_id'],record['start_ms'],record['end_ms']))
            repaired+=cursor.rowcount
    return repaired


async def recording_links_lifecycle(app):
    async def monitor():
        failure_reported=False
        while True:
            try:
                await reconcile_recordings(app)
                failure_reported=False
            except asyncio.CancelledError:
                raise
            except Exception:
                if not failure_reported:
                    print('Protect recording links unavailable; retrying in the background.',flush=True)
                    failure_reported=True
            await asyncio.sleep(RECORDING_SYNC_INTERVAL)
    task=asyncio.create_task(monitor())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass


async def thumbnail_lifecycle(app):
    """Fill attention frames in a separate CPU process, including legacy events."""
    if os.environ.get('LITTERLOG_THUMBNAIL_WORKER', '1') == '0':
        yield
        return
    async def monitor():
        while True:
            process=None
            try:
                with (state_root(ROOT)/'thumbnail-worker.log').open('ab',buffering=0) as log:
                    process=await asyncio.create_subprocess_exec(
                        sys.executable,'-u',str(ROOT/'backfill_thumbnails.py'),
                        '--watch','--database',str(DB),stdin=asyncio.subprocess.DEVNULL,
                        stdout=log,stderr=log)
                    result=await process.wait()
                # A second server may already own the shared lock. Retry later
                # so this server can take over if that owner shuts down.
                if result!=0: print('Attention thumbnail worker stopped; retrying in 30 seconds.',flush=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                print('Attention thumbnail worker unavailable; retrying in 30 seconds.',flush=True)
            finally:
                if process is not None and process.returncode is None:
                    process.terminate()
                    try: await asyncio.wait_for(process.wait(),5)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
            await asyncio.sleep(30)
    task=asyncio.create_task(monitor())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass


def prune_clip_cache(app,requested):
    limit=app.get('clip_cache_limit',2*1024**3)
    if limit<=0: return
    # Do not evict files being prepared for another waiting playback request.
    protected=set(app['clip_tasks']) | {requested.name}
    files=sorted((p for p in app['clips'].glob('*.mp4')
                  if re.fullmatch(r'[a-f0-9]{64}\.mp4',p.name)),key=lambda p:p.stat().st_mtime)
    size=sum(p.stat().st_size for p in files)
    for file in files:
        if size<=limit: break
        if file.name not in protected:
            size-=file.stat().st_size
            file.unlink()


async def prepare_clip(app,row,destination):
    # Limit new NVR exports/remux jobs, while cached clips bypass this queue.
    async with app['clip_slots']:
        clip_storage.ensure_available(destination.parent)
        if not destination.exists():
            await asyncio.to_thread(app['protect'].export,*row,destination)
        else:
            await asyncio.to_thread(ensure_faststart,destination)
        destination.touch()
        prune_clip_cache(app,destination)


def start_clip_job(tasks, name, operation):
    job = asyncio.create_task(operation)
    tasks[name] = job

    def finished(task):
        if tasks.get(name) is task:
            tasks.pop(name, None)
        # Retrieve an exception even when every waiting phone cancelled.
        if not task.cancelled():
            task.exception()

    job.add_done_callback(finished)
    return job


async def clip_jobs_lifecycle(app):
    yield
    # Shielded exports may outlive an individual phone request; finish them
    # before shutting down to avoid abandoned partial files or duplicate jobs.
    await asyncio.gather(*list(app['clip_tasks'].values()),return_exceptions=True)


async def playback(request):
    eid=request.match_info['event_id']
    archive=request.app.get('history_archive')
    with history_store(archive) as c:
        row=c.execute('SELECT event_id,camera_id,start_ms,end_ms FROM history_recordings WHERE event_id=?',(eid,)).fetchone()
        if row is None: require_archived_history(c,archive)
    if row is None: raise web.HTTPNotFound()
    name=hashlib.sha256(eid.encode()).hexdigest()+'.mp4'
    tasks=request.app['clip_tasks']
    job=tasks.get(name)
    clip_storage.ensure_available(request.app['clips'])
    request.app['clips'].mkdir(parents=True,exist_ok=True)
    destination=request.app['clips']/name
    if job is None:
        if destination.exists() and faststart_ready(destination):
            destination.touch()
        else:
            job = start_clip_job(tasks, name, prepare_clip(request.app, row, destination))
    if job is not None:
        await asyncio.shield(job)
    return web.json_response({'url':'/media/clips/'+name+'?token='+quote(request.app['token'],safe='')})

async def attention_thumbnail(request):
    name=request.match_info['name']
    if not re.fullmatch(r'[a-f0-9]{64}\.jpg',name): raise web.HTTPNotFound()
    folder = state_root(ROOT) / 'attention-cache'
    headers = {'Content-Type':'image/jpeg','Cache-Control':'private,max-age=3600'}
    file = folder / name
    if not file.exists(): raise web.HTTPNotFound()
    return web.FileResponse(file, headers=headers)


async def recorded_clip(request):
    name=request.match_info['name']
    if not re.fullmatch(r'[a-f0-9]{64}\.mp4',name): raise web.HTTPNotFound()
    clip_storage.ensure_available(request.app['clips'])
    file=request.app['clips']/name
    if not file.exists(): raise web.HTTPNotFound()
    # aiohttp FileResponse implements byte ranges for AVPlayer seeking.
    return web.FileResponse(file,headers={'Content-Type':'video/mp4','Cache-Control':'no-store'})

async def media(request):
    name=request.match_info['name']
    if not re.fullmatch(r'live\.m3u8|part[0-9]{9}\.ts',name): raise web.HTTPNotFound()
    directory=request.app.get('media_dir')
    if directory is not None:
        clip_storage.ensure_available(directory)
    if directory is None or not (directory/name).exists(): raise web.HTTPNotFound()
    if name.endswith('.m3u8'):
        lines=(directory/name).read_text().splitlines()
        body='\n'.join(line+'?token='+quote(request.app['token'],safe='') if line and not line.startswith('#') else line for line in lines)+'\n'
        return web.Response(text=body,content_type='application/vnd.apple.mpegurl',headers={'Cache-Control':'no-store'})
    return web.FileResponse(directory/name,headers={'Content-Type':'video/mp2t','Cache-Control':'no-store'})

def create_app(token, email, password):
    app=web.Application(middlewares=[auth],client_max_size=16384)
    app.update(token=token,email=email,password=password)
    app['protect']=ProtectService()
    app['thumbnail_index']={}
    app['history_retention_months']=int(os.environ.get('LITTERLOG_LOCAL_HISTORY_MONTHS','0'))
    if app['history_retention_months']<0:
        raise ValueError('LITTERLOG_LOCAL_HISTORY_MONTHS must be nonnegative')
    app['clips']=clip_storage.configured_root()/'clip-cache'
    app['clip_cache_limit']=int(float(os.environ.get('LITTERLOG_CLIP_CACHE_GB','0'))*1024**3)
    if app['clip_cache_limit']<0: raise ValueError('LITTERLOG_CLIP_CACHE_GB must be nonnegative')
    app['clip_tasks']={}
    app['clip_slots'] = asyncio.Semaphore(2)
    app.cleanup_ctx.append(clip_jobs_lifecycle)
    app.cleanup_ctx.append(history_archive_lifecycle)
    app.cleanup_ctx.append(lifecycle)
    app.cleanup_ctx.append(recording_links_lifecycle)
    app.cleanup_ctx.append(thumbnail_lifecycle)
    app.cleanup_ctx.append(media_lifecycle)
    app.router.add_get('/api/camera',camera)
    app.router.add_get('/media/snapshot.jpg',live_snapshot)
    app.router.add_get('/media/clips/{name}',recorded_clip)
    app.router.add_get('/media/thumbs/{name}',attention_thumbnail)
    app.router.add_post('/api/recordings/import',import_recordings)
    app.router.add_get('/api/recordings/{event_id}/play',playback)
    app.router.add_get('/media/{name}',media)
    app.router.add_get('/api/robot',status)
    app.router.add_post('/api/robot/command',command)
    app.router.add_get('/api/events',events)
    app.router.add_get('/api/weights',weights)
    app.router.add_get('/api/cat-activity',cat_activity)
    app.router.add_post('/api/events/{id}/override',override_prediction)
    app.router.add_get('/api/events/known',known_events)
    app.router.add_post('/api/events',events)
    return app

if __name__=='__main__':
    token=pairing_token()
    if len(token)<24: raise SystemExit('Choose a token of at least 24 characters.')
    email=os.environ.get('WHISKER_EMAIL') or os.environ.get('LITTERLOG_WHISKER_EMAIL') or input('Whisker email: ')
    password=os.environ.get('WHISKER_PASSWORD') or os.environ.get('LITTERLOG_WHISKER_PASSWORD') or getpass.getpass('Whisker password: ')
    tls=None
    if os.environ.get('LITTERLOG_TLS_CERT'):
        tls=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(os.environ['LITTERLOG_TLS_CERT'],os.environ['LITTERLOG_TLS_KEY'])
    host=os.environ.get('LITTERLOG_BIND','127.0.0.1')
    if host not in ('127.0.0.1','localhost','::1') and tls is None:
        raise SystemExit('Network access requires LITTERLOG_TLS_CERT and LITTERLOG_TLS_KEY, or an HTTPS reverse proxy to loopback.')
    web.run_app(create_app(token,email,password),host=host,port=int(os.environ.get('LITTERLOG_PORT',os.environ.get('PORT','8767'))),ssl_context=tls,access_log=None)
