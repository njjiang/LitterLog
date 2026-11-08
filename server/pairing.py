"""User-authorized local pairing-token persistence; never stores account passwords."""
import getpass
import os
import re
import shlex
from pathlib import Path

START='# >>> LitterLog app pairing >>>'
END='# <<< LitterLog app pairing <<<'

def pairing_token():
    existing=os.environ.get('LITTERLOG_APP_TOKEN','')
    if existing:
        if len(existing)<24: raise SystemExit('LITTERLOG_APP_TOKEN must have at least 24 characters.')
        return existing
    token=getpass.getpass('Choose app pairing token (24+ characters; saved in ~/.zshrc): ')
    if len(token)<24 or '\n' in token or '\r' in token:
        raise SystemExit('Choose a single-line token with at least 24 characters.')
    path=(Path.home()/'.zshrc').resolve()
    text=path.read_text() if path.exists() else ''
    block=START+'\nexport LITTERLOG_APP_TOKEN='+shlex.quote(token)+'\n'+END
    pattern=re.compile(re.escape(START)+r'.*?'+re.escape(END),re.S)
    if pattern.search(text): text=pattern.sub(lambda _:block,text)
    else: text=text.rstrip()+'\n\n'+block+'\n'
    try:
        with path.open('w') as handle:
            os.fchmod(handle.fileno(),0o600)
            handle.write(text)
    except OSError:
        raise SystemExit('Could not save token to ~/.zshrc. No server started. Check file permissions.')
    os.environ['LITTERLOG_APP_TOKEN']=token
    print('App pairing token saved in ~/.zshrc (value hidden).')
    return token
