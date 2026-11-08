"""Non-interactive container start; load mounted secrets without logging values."""
import os
import sys
from pathlib import Path

SECRET_NAMES = (
    'LITTERLOG_APP_TOKEN', 'WHISKER_EMAIL', 'WHISKER_PASSWORD',
    'LITTERLOG_NVR_USER', 'LITTERLOG_NVR_PASS',
)


def load_secret_environment():
    for name in SECRET_NAMES:
        path = os.environ.get(name + '_FILE')
        if not path:
            continue
        if os.environ.get(name):
            raise ValueError(f'Set either {name} or {name}_FILE, not both.')
        value = Path(path).read_text().rstrip('\r\n')
        if not value or '\n' in value or '\r' in value:
            raise ValueError(f'{name}_FILE must contain one nonempty line.')
        os.environ[name] = value


def load_protect_routing():
    config = os.environ.get('LITTERLOG_PROTECT_ROUTING_DIR')
    if not config:
        return
    try:
        for filename, variable, is_required in (
            ('protect-host', 'LITTERLOG_NVR_HOST', True),
            ('protect-tls-name', 'LITTERLOG_NVR_TLS_NAME', False),
        ):
            value = (Path(config) / filename).read_text().rstrip('\r\n')
            if (is_required and not value) or any(char in value for char in '\r\n\0'):
                raise ValueError()
            os.environ[variable] = value
    except (OSError, ValueError):
        raise ValueError('Protect address files are missing or invalid.') from None


def required(name):
    value = os.environ.get(name, '')
    if not value:
        raise ValueError(f'{name} must be configured.')
    return value


def required_file(name):
    path = Path(required(name))
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f'{name} must point to a readable file.')


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    role = args.pop(0) if args else 'api'
    if role not in ('api', 'worker', 'thumbnails'):
        raise ValueError('Choose api, worker, or thumbnails.')
    load_protect_routing()
    load_secret_environment()
    if len(required('LITTERLOG_APP_TOKEN')) < 24:
        raise ValueError('LITTERLOG_APP_TOKEN must contain at least 24 characters.')
    for name in ('LITTERLOG_NVR_HOST', 'LITTERLOG_NVR_USER', 'LITTERLOG_NVR_PASS'):
        required(name)
    required_file('LITTERLOG_NVR_CA_CERT')
    state = Path(required('LITTERLOG_STATE_DIR'))
    if not state.is_dir() or not os.access(state, os.W_OK):
        raise ValueError('The local state volume must exist and be writable by the container user.')
    media = Path(os.environ.get('LITTERLOG_MEDIA_ROOT', str(state / 'media')))
    media.mkdir(parents=True, exist_ok=True)
    if not media.is_dir() or not os.access(media, os.W_OK):
        raise ValueError('The media directory must exist and be writable by the container user.')
    if role == 'api':
        required('WHISKER_EMAIL')
        required('WHISKER_PASSWORD')
        required_file('LITTERLOG_TLS_CERT')
        required_file('LITTERLOG_TLS_KEY')
        required_file('LITTERLOG_SERVER_CA_CERT')
        command = [sys.executable, '-u', '/app/server.py', *args]
    else:
        model = Path(required('LITTERLOG_MODEL_DIR'))
        checkpoint = os.environ.get('LITTERLOG_CHECKPOINT', 'video_24f_balanced')
        if not (model / 'labels.txt').is_file() or not (model / 'checkpoints' / checkpoint / 'best.npz').is_file():
            raise ValueError('Mount labels.txt and the selected checkpoint under LITTERLOG_MODEL_DIR.')
        if role == 'worker':
            server_ca = ('LITTERLOG_SERVER_CA_CERT' if 'LITTERLOG_SERVER_CA_CERT' in os.environ
                         else 'REQUESTS_CA_BUNDLE')
            required_file(server_ca)
            os.environ['LITTERLOG_SERVER_CA_CERT'] = required(server_ca)
            server = required('LITTERLOG_SERVER_URL')
            if not server.startswith('https://'):
                raise ValueError('The container worker requires an HTTPS server URL.')
            command = [sys.executable, '-u', '/app/worker.py', '--runtime', 'cpu',
                       '--server', server, '--camera', required('LITTERLOG_CAMERA_NAME'),
                       '--model-dir', str(model), '--checkpoint', checkpoint, *args]
        else:
            command = [sys.executable, '-u', '/app/backfill_thumbnails.py', '--watch',
                       '--model-dir', str(model), '--checkpoint', checkpoint, *args]
    os.execv(sys.executable, command)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as error:
        message = str(error) if isinstance(error, ValueError) else 'A required file or mounted volume could not be read.'
        raise SystemExit('LitterLog container configuration: ' + message)
