import os
from databoard import db
import databoard.config as config


def recreate_db():
    """Initialisation of a test database."""
    db.session.close()
    db.drop_all()
    db.create_all()
    print(db)


def deploy():
    if os.getenv('DATABOARD_TEST'):
        os.system('rm -rf ' + config.local_test_deployment_path)
        os.makedirs(config.local_test_deployment_path)
        if not os.path.isdir(config.deployment_path):
            os.makedirs(config.deployment_path)
        os.system('rsync -rultv fabfile.py ' + config.deployment_path)
        os.makedirs(config.ramp_kits_path)
        os.makedirs(config.ramp_data_path)
        os.makedirs(config.submissions_path)
        recreate_db()
