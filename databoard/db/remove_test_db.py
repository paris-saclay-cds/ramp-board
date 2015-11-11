import os
import databoard.config as config
from databoard.db.model import db


def remove_test_db():
    if not os.path.exists(config.db_path):
        os.mkdir(config.db_path)
    try:
        os.remove(config.db_f_name)
    except OSError:
        pass


def recreate_test_db():
    remove_test_db()
    db.create_all()
