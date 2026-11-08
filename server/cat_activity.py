"""Estimated bathroom visits from corrected, cat-attributed waste detections."""
from datetime import datetime, timezone


def visits(events):
    rows = sorted(events, key=lambda e: (datetime.fromisoformat(e['time'].replace('Z', '+00:00')), e['id']))
    result = []
    current = None
    seen = set()
    for event in rows:
        cat, label = event.get('cat'), event['label'].lower()
        # A different identified cat interrupts a possible duplicate group.
        if current and cat in ('cat1', 'cat2') and current['cat'] != cat:
            current = None
        if cat not in ('cat1', 'cat2') or label not in ('pee', 'poo', 'both'):
            continue
        key = event.get('event_id') or event['id']
        if key in seen:
            continue
        seen.add(key)
        time = datetime.fromisoformat(event['time'].replace('Z', '+00:00')).timestamp()
        if current is None or time - current['_start'] > 600:
            current = dict(id=event['id'], cat=cat, time=datetime.fromtimestamp(time, timezone.utc).isoformat(), pee=False, poo=False, _start=time)
            result.append(current)
        current['pee'] |= label in ('pee', 'both')
        current['poo'] |= label in ('poo', 'both')
    return [{k: v for k, v in row.items() if k != '_start'} for row in result]
