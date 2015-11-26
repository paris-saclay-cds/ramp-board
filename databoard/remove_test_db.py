import os
from databoard.config import db_path, db_f_name
from databoard import db


def remove_test_db():
    if not os.path.exists(db_path):
        os.mkdir(db_path)
    try:
        os.remove(db_f_name)
    except OSError:
        pass


def recreate_test_db():
    remove_test_db()
    db.create_all()
