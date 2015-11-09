import os
from databoard.config import set_engine_and_session, get_session, get_engine, db_path

if not os.path.exists(db_path):
    os.mkdir(db_path)
db_f_name = os.path.join(db_path, 'test.db')
try:
    os.remove(db_f_name)
except OSError:
    pass
set_engine_and_session('sqlite:///' + db_f_name, echo=False)

session = get_session()
engine = get_engine()

from databoard.db.model_base import DBBase, NameClashError
import databoard.db.teams as teams
import databoard.db.users as users
import databoard.db.submissions as submissions

DBBase.metadata.create_all(engine)
