"""Conservative timestamp attribution, preserving raw model/Whisker data."""
from datetime import datetime

CATS={'cat1','cat2'}

def seconds(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()

def identify(events):
    """Manual/video identities are anchors. Waste uses the last recent video ID."""
    rows=sorted((dict(e) for e in events),key=lambda e:(seconds(e['time']),e['id']))
    anchors=[]
    for e in rows:
        if e.get('manual_cat') is not None:
            e['cat']=e['manual_cat']
            e['attribution']='confirmed' if e['cat'] else 'unassigned'
        elif e.get('cat') in CATS and e.get('confidence',0)>=.8:
            e['attribution']='video prediction'
        else:
            e['cat']=''
            e['attribution']='unassigned'
        if e['cat'] in CATS and e['attribution'] in ('confirmed','video prediction'):
            anchors.append(e)
        elif e['label'] in ('pee','poo','both') and e.get('manual_cat') is None:
            recent=[a for a in anchors if 0<=seconds(e['time'])-seconds(a['time'])<=30*60]
            if recent:
                e['cat']=recent[-1]['cat']
                e['attribution']='last video ID by timestamp'
                e['identity_event_id']=recent[-1]['id']
    return rows

def match_weight(reading,events):
    """Match recordings independently; a covering clip has first claim on identity."""
    timestamp=seconds(reading['time'])
    result=dict(reading,cat='',attribution='unassigned',identity_event_id=None,
                video_event_id=None,video_match=None)
    def interval(e):
        start,end=e.get('start_ms'),e.get('end_ms')
        if start is None or end is None or end<=start: return None
        return start/1000,end/1000
    def covers(e):
        span=interval(e)
        return span is not None and span[0]<=timestamp<=span[1]
    def distance(e):
        span=interval(e)
        return max(span[0]-timestamp,timestamp-span[1],0) if span else abs(seconds(e['time'])-timestamp)
    covering=[e for e in events if e.get('has_video') and covers(e)]
    nearby=[e for e in events if e.get('has_video') and distance(e)<=180]
    videos=covering or nearby
    if videos:
        video=min(videos,key=lambda e:(distance(e),abs((interval(e) or (0,seconds(e['time'])))[1]-timestamp),e['id']))
        result.update(video_event_id=video['id'],video_match='covers weigh-in' if covering else 'near weigh-in')
    # Do not substitute an adjacent cat when the actual recording is uncertain.
    pool=covering if covering else [e for e in events if abs(seconds(e['time'])-timestamp)<=180]
    if covering and any(e.get('manual_cat')=='' for e in covering):
        return result
    candidates=[e for e in pool if e.get('cat') in CATS and e.get('attribution') in ('confirmed','video prediction')]
    if not candidates: return result
    confirmed=[e for e in candidates if e['attribution']=='confirmed']
    if confirmed: candidates=confirmed
    if len({e['cat'] for e in candidates})>1:
        result['attribution']='ambiguous video identities'
        return result
    winner=min(candidates,key=lambda e:(abs(seconds(e['time'])-timestamp),e['id']))
    result.update(cat=winner['cat'],attribution='confirmed video tag' if winner['attribution']=='confirmed' else 'video prediction by timestamp',identity_event_id=winner['id'])
    return result
