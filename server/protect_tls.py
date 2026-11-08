"""Explicit verified TLS identity for a Protect console reached via a LAN alias."""
from requests.adapters import HTTPAdapter

class ProtectTLSAdapter(HTTPAdapter):
    def __init__(self,tls_name):
        self.tls_name=tls_name
        super().__init__()
    def init_poolmanager(self,connections,maxsize,block=False,**kwargs):
        kwargs['assert_hostname']=self.tls_name
        kwargs['server_hostname']=self.tls_name
        return super().init_poolmanager(connections,maxsize,block=block,**kwargs)
    def proxy_manager_for(self,*args,**kwargs):
        raise ValueError('Proxies are not supported for the local Protect TLS connection.')
