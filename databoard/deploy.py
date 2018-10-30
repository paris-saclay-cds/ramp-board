import os

from databoard import (db, deployment_path, ramp_config, ramp_data_path,
                       ramp_kits_path)


def recreate_db():
    """Initialisation of a test database."""
    db.session.close()
    db.drop_all()
    db.create_all()
    print(db)


def deploy():
    if os.getenv('DATABOARD_STAGE') in ['TEST', 'TESTING']:
        os.unlink(deployment_path)
        os.makedirs(deployment_path)
        os.system('rsync -rultv fabfile.py {}'.format(deployment_path))
        os.makedirs(ramp_kits_path)
        os.makedirs(ramp_data_path)
        os.makedirs(
            os.path.join(deployment_path, ramp_config['submissions_dir'])
        )
        recreate_db()
