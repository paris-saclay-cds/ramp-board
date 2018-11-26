import os
import subprocess
from shutil import rmtree

from . import db
from . import deployment_path
from . import ramp_config


def deploy_test_database():
    """Deploy the RAMP model for testing purpose.

    The different settings should be set as environment variables. The
    environment variables to set are:

    * DATABOARD_STAGE: stage of the database. Need to be `'TESTING'` to use
      this function;
    * DATABOARD_USER: database login;
    * DATABOARD_PASSWORD: database password;
    * DATABOARD_DB_URL_TEST: database URL.
    """
    if os.getenv('DATABOARD_STAGE') in ['TEST', 'TESTING']:
        rmtree(deployment_path, ignore_errors=True)
        os.makedirs(deployment_path)
        subprocess.run(["rsync", "-rultv", 'fabfile.py', deployment_path])
        os.makedirs(ramp_config['ramp_kits_path'])
        os.makedirs(ramp_config['ramp_data_path'])
        os.makedirs(ramp_config['ramp_submissions_path'])
        # create the empty database
        db.session.close()
        db.drop_all()
        db.create_all()
    else:
        raise AttributeError('DATABOARD_STAGE should be set to TESTING for '
                             '`deploy` to work')
