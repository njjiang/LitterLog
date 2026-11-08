#!/usr/bin/env python3
"""
Download motion event video clips from UniFi Protect for dataset labeling.

Usage:
    python download_events.py --host <nvr-host> --user admin --pass secret
    python download_events.py --host <nvr-host> --user admin --pass secret --days 30
    python download_events.py --host <nvr-host> --user admin --pass secret --camera "Litter Box"

Output structure:
    unifi_data/
        <camera-name>/
            <timestamp>_<event-id>.mp4
    events_meta.jsonl   # one line per event with metadata for labeling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = Path(__file__).resolve().parent / "unifi_data"


class UnifiProtect:
    def __init__(self, host: str, verify_ssl: bool = False):
        self.base = f"https://{host}"
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.bootstrap = None

    def login(self, username: str, password: str) -> None:
        resp = self.session.post(f"{self.base}/api/auth/login", json={
            "username": username,
            "password": password,
            "rememberMe": True,
        }, timeout=15)
        resp.raise_for_status()
        csrf = resp.headers.get("X-Updated-CSRF-Token") or resp.headers.get("x-csrf-token")
        if csrf:
            self.session.headers["x-csrf-token"] = csrf
        print(f"[+] Logged in as {username}")

    def get_bootstrap(self) -> dict:
        if self.bootstrap is None:
            resp = self.session.get(f"{self.base}/proxy/protect/api/bootstrap", timeout=15)
            resp.raise_for_status()
            self.bootstrap = resp.json()
        return self.bootstrap

    def cameras(self) -> dict[str, str]:
        bs = self.get_bootstrap()
        return {c["id"]: c["name"] for c in bs.get("cameras", [])}

    def events(
        self,
        start_ms: int,
        end_ms: int,
        camera_ids: list[str] | None = None,
        types: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict]:
        params: dict = {"start": start_ms, "end": end_ms, "limit": limit}
        if camera_ids:
            params["cameras"] = ",".join(camera_ids)
        if types:
            params["types"] = ",".join(types)

        all_events: list[dict] = []
        while True:
            resp = self.session.get(
                f"{self.base}/proxy/protect/api/events",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_events.extend(batch)
            if len(batch) < limit:
                break
            params["start"] = batch[-1]["start"] + 1
        return all_events

    def download_clip(self, camera_id: str, start_ms: int, end_ms: int) -> bytes | None:
        """Download the video clip for a time range from a camera."""
        resp = self.session.get(
            f"{self.base}/proxy/protect/api/video/export",
            params={
                "camera": camera_id,
                "start": start_ms,
                "end": end_ms,
            },
            timeout=120,
            stream=True,
        )
        if resp.status_code != 200:
            return None
        ct = resp.headers.get("content-type", "")
        if "video" not in ct and "octet-stream" not in ct:
            return None
        return resp.content


def slugify(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").lower()


def save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description="Download UniFi Protect event clips")
    parser.add_argument("--host", required=True, help="NVR IP or hostname")
    parser.add_argument("--user", required=True, help="Username")
    parser.add_argument("--pass", dest="password", required=True, help="Password")
    parser.add_argument("--days", type=int, default=90, help="How many days back to fetch (default 90)")
    parser.add_argument("--camera", help="Filter by camera name (substring match, case-insensitive)")
    parser.add_argument(
        "--types",
        default=None,
        help="Event types, comma-separated. Omit to fetch all types.",
    )
    parser.add_argument("--ssl", action="store_true", help="Verify SSL certificate")
    parser.add_argument("--output-dir", type=str, default=None, help="Override OUTPUT_DIR (default: unifi_data)")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    protect = UnifiProtect(args.host, verify_ssl=args.ssl)
    protect.login(args.user, args.password)

    cam_map = protect.cameras()
    print(f"[+] Found {len(cam_map)} cameras: {list(cam_map.values())}")

    target_camera_ids: list[str] | None = None
    if args.camera:
        target_camera_ids = [
            cid for cid, name in cam_map.items()
            if args.camera.lower() in name.lower()
        ]
        if not target_camera_ids:
            print(f"[!] No cameras match '{args.camera}'. Available: {list(cam_map.values())}")
            sys.exit(1)
        print(f"[+] Filtering to: {[cam_map[i] for i in target_camera_ids]}")

    now = datetime.now(tz=timezone.utc)
    start_ms = int((now - timedelta(days=args.days)).timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)
    types = [t.strip() for t in args.types.split(",")] if args.types else None

    print(f"[+] Fetching events from the last {args.days} days …")
    events = protect.events(start_ms, end_ms, camera_ids=target_camera_ids, types=types)
    print(f"[+] Got {len(events)} events")

    meta_path = OUTPUT_DIR / "events_meta.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing metadata and preserve labels already assigned
    existing: dict[str, dict] = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                existing[e["id"]] = e
            except Exception:
                pass

    downloaded = skipped = errors = 0

    with meta_path.open("w") as meta_f:
        for i, ev in enumerate(events, 1):
            ev_id = ev["id"]
            cam_id = ev.get("camera", "")
            cam_name = cam_map.get(cam_id, cam_id or "unknown")
            cam_slug = slugify(cam_name)
            ev_start = ev["start"]
            ev_end = ev.get("end") or (ev_start + 60_000)  # fallback: 60s clip

            ts = datetime.fromtimestamp(ev_start / 1000, tz=timezone.utc)
            ts_str = ts.strftime("%Y%m%d_%H%M%S")

            clip_path = OUTPUT_DIR / cam_slug / f"{ts_str}_{ev_id}.mp4"

            if clip_path.exists():
                skipped += 1
            else:
                data = protect.download_clip(cam_id, ev_start, ev_end)
                if data:
                    save(clip_path, data)
                    downloaded += 1
                else:
                    errors += 1

            # Preserve any label already assigned in a previous run
            existing_label = existing.get(ev_id, {}).get("label", None)
            meta_f.write(json.dumps({
                "id": ev_id,
                "camera": cam_name,
                "start": ev_start,
                "end": ev_end,
                "type": ev.get("type"),
                "smartDetectTypes": ev.get("smartDetectTypes", []),
                "clip": str(clip_path.relative_to(OUTPUT_DIR)),
                "label": existing_label,
            }) + "\n")

            if i % 10 == 0 or i == len(events):
                print(f"  {i}/{len(events)}  downloaded={downloaded}  skipped={skipped}  errors={errors}")
            time.sleep(0.1)

    print(f"\n[+] Done. Clips saved to: {OUTPUT_DIR}")
    print(f"    Downloaded: {downloaded}  |  Already existed: {skipped}  |  Errors: {errors}")
    print(f"    Metadata: {meta_path}")
    print(f"\n    Next: watch each clip and fill in the 'label' field in events_meta.jsonl")


if __name__ == "__main__":
    main()
