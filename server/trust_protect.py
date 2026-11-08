"""Save a specific Protect certificate only after independent fingerprint verification."""
import hashlib
import os
import socket
import ssl
from pathlib import Path

host=os.environ.get('LITTERLOG_NVR_HOST') or input('Protect hostname: ').strip()
if '://' in host or '/' in host:
    raise SystemExit('Use a hostname only, without https:// or paths.')
context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
# Retrieval only: no credentials or API requests are sent over this connection.
context.check_hostname=False
context.verify_mode=ssl.CERT_NONE
with socket.create_connection((host,443),timeout=10) as connection:
    with context.wrap_socket(connection,server_hostname=host) as peer:
        certificate=peer.getpeercert(binary_form=True)
fingerprint=hashlib.sha256(certificate).hexdigest().upper()
print('Protect server certificate SHA-256 fingerprint:')
print(':'.join(fingerprint[i:i+2] for i in range(0,len(fingerprint),2)))
print('\nCompare this with the SHA-256 fingerprint in the certificate details of your known Protect console.')
print('Do not continue merely because the hostname looks familiar.')
entered=input('Paste the independently verified fingerprint, or press Return to cancel: ')
entered=entered.replace(':','').replace(' ','').strip().upper()
if entered != fingerprint:
    raise SystemExit('Certificate not saved. Fingerprints did not match or verification was cancelled.')
folder=Path(__file__).resolve().parent/'trusted-protect'
folder.mkdir(mode=0o700,exist_ok=True)
path=folder/(hashlib.sha256(host.encode()).hexdigest()+'.pem')
path.write_text(ssl.DER_cert_to_PEM_cert(certificate))
path.chmod(0o600)
print('Certificate saved for this Protect host. Run the video-model launcher again.')
