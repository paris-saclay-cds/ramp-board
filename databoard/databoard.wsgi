import os
import sys
import site

site.addsitedir('/usr/local/lib/python2.7/site-packages')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

def application(environ, start_response):
    from databoard import app as _application
    ENV_VAR = ['DATABOARD_DB_URL']
    for kk in ENV_VAR:
        os.environ[kk] = environ.get(kk, '')
    return _application(environ, start_response)
