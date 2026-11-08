"""Isolated image checks: temporary fixtures only, no robot or camera connections.

Run explicitly with `python deploy/smoke_test.py`. This does not use production
credentials, storage, or model weights. Real checkpoint parity is tested separately.
"""
import asyncio
import hashlib
import ipaddress
import json
import os
import secrets
import socket
import ssl
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))


def temporary_certificate(folder):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'LitterLog image test')])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
                   .public_key(key.public_key()).serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
                   .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
                   .add_extension(x509.SubjectAlternativeName([
                       x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
                   .sign(key, hashes.SHA256()))
    cert_path, key_path = folder / 'test-cert.pem', folder / 'test-key.pem'
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                         serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    key_path.chmod(0o600)
    return cert_path, key_path


async def api_check(folder):
    import aiohttp
    from aiohttp import web
    import server
    from healthcheck import main as healthcheck
    token = secrets.token_urlsafe(32)
    app = server.create_app(token, 'fixture@example.invalid', 'fixture-only')
    # Exercise real HTTP handlers without launching external-service monitors.
    app.cleanup_ctx.clear()
    cert, key = temporary_certificate(folder)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    client_tls = ssl.create_default_context(cafile=str(cert))
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.setblocking(False)
    try:
        await web.SockSite(runner, listener, ssl_context=tls).start()
        base = f'https://127.0.0.1:{port}'
        headers = {'Authorization': 'Bearer ' + token}
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=client_tls),
                                         timeout=aiohttp.ClientTimeout(total=15)) as client:
            async with client.get(base + '/api/events') as response:
                assert response.status == 401, 'Unauthenticated request was accepted'
            async with client.get(base + '/api/events', headers=headers) as response:
                assert response.status == 200 and await response.json() == []
            event = {'event_id': 'smoke-clip', 'time': datetime.now(timezone.utc).isoformat(),
                     'label': 'cat1', 'confidence': 0.9, 'cat': 'cat1',
                     'camera_id': 'fixture-camera', 'start_ms': 1000, 'end_ms': 2000}
            async with client.post(base + '/api/events', headers=headers, json=event) as response:
                assert response.status == 201
            async with client.get(base + '/api/events', headers=headers) as response:
                events = await response.json()
                assert response.status == 200 and events[0]['event_id'] == 'smoke-clip'
            async with client.get(base + '/api/recordings/smoke-clip/play', headers=headers) as response:
                assert response.status == 200
                playback = (await response.json())['url']
            async with client.get(base + playback, headers={'Range': 'bytes=0-31'}) as response:
                assert response.status == 206 and len(await response.read()) == 32
            async with client.get(base + '/api/weights', headers=headers) as response:
                assert response.status == 200 and await response.json() == []
        os.environ.update(LITTERLOG_BIND='127.0.0.1', LITTERLOG_PORT=str(port),
                          LITTERLOG_APP_TOKEN=token, LITTERLOG_SERVER_CA_CERT=str(cert))
        assert await asyncio.to_thread(healthcheck) == 0
    finally:
        await runner.cleanup()
        listener.close()
    return ['verified_https', 'authentication', 'event_write_read', 'cached_video_byte_ranges',
            'weight_endpoint', 'container_healthcheck']


def main():
    # No inherited service configuration is used, even when invoked manually.
    for name in list(os.environ):
        if name.startswith(('LITTERLOG_', 'WHISKER_')) and name != 'LITTERLOG_CPU_THREADS':
            os.environ.pop(name)
    with tempfile.TemporaryDirectory(prefix='litterlog-image-smoke-') as temporary:
        folder = Path(temporary)
        media = folder / 'media'
        clips = media / 'clip-cache'
        clips.mkdir(parents=True)
        os.environ.update(LITTERLOG_STATE_DIR=str(folder), LITTERLOG_DB=str(folder / 'events.sqlite3'),
                          LITTERLOG_MEDIA_ROOT=str(media), LITTERLOG_THUMBNAIL_WORKER='0')
        import av
        import imageio_ffmpeg
        import torch
        import torchvision
        import worker
        import backfill_thumbnails
        from cpu_infer import VideoModel, configure_cpu_threads
        threads = configure_cpu_threads()
        model = VideoModel(5).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 2, 224, 224, 3))
        assert output.shape == (1, 5) and torch.isfinite(output).all()
        assert torch.allclose(output.sum(), torch.tensor(1.0))
        clip = clips / (hashlib.sha256(b'smoke-clip').hexdigest() + '.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-nostdin', '-y', '-f', 'lavfi', '-i', 'testsrc2=size=64x64:rate=5:duration=1',
                        '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(clip)], check=True, timeout=30)
        with av.open(str(clip)) as video:
            assert sum(1 for _ in video.decode(video=0)) == 5
        checks = asyncio.run(api_check(folder))
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('LITTERLOG_', 'WHISKER_'))}
        stopped = subprocess.run([sys.executable, str(ROOT / 'deploy/container_entrypoint.py'), 'api'],
                                 env=environment, capture_output=True, text=True, timeout=10)
        assert stopped.returncode != 0 and 'LITTERLOG_APP_TOKEN must be configured' in stopped.stderr
        checks += ['cpu_forward', 'ffmpeg_encode', 'pyav_decode', 'worker_import',
                   'thumbnail_worker_import', 'missing_configuration_refused']
        print('LITTERLOG_IMAGE_SMOKE_RESULT=' + json.dumps({
            'status': 'passed', 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cpu_threads': threads, 'checks': checks,
            'scope': 'isolated fixtures; no camera, robot, NAS, or trained-checkpoint connections'}), flush=True)


if __name__ == '__main__':
    main()
