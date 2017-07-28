import os
import sys
import site

site.addsitedir('/home/ubuntu/miniconda/lib/python2.7/site-packages')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

def application(environ, start_response):
    ENV_VAR = ['DATABOARD_DB_URL']
    for kk in ENV_VAR:
        os.environ[kk] = environ.get(kk, '') 
    from databoard import app as _application
    return _application(environ, start_response)
