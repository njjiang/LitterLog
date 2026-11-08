"""Check local API availability without contacting or commanding the robot."""
import ipaddress
import os
import ssl
import sys
import urllib.request

from container_entrypoint import load_secret_environment


def health_url():
    address = ipaddress.ip_address(os.environ.get('LITTERLOG_BIND', '0.0.0.0'))
    if address.is_unspecified:
        address = ipaddress.ip_address('127.0.0.1' if address.version == 4 else '::1')
    host = f'[{address}]' if address.version == 6 else str(address)
    return f"https://{host}:{os.environ.get('LITTERLOG_PORT', '8443')}/api/events"


def main():
    load_secret_environment()
    context = ssl.create_default_context(cafile=os.environ['LITTERLOG_SERVER_CA_CERT'])
    request = urllib.request.Request(
        health_url(),
        headers={'Authorization': 'Bearer ' + os.environ['LITTERLOG_APP_TOKEN']},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context))
    with opener.open(request, timeout=8) as response:
        if response.status != 200:
            return 1
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        # Docker persists healthcheck output; keep credentials and URLs out of it.
        print('LitterLog API health check failed.')
        sys.exit(1)
