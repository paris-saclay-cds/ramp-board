import os
from ..config import set_engine_and_session, db_path

if not os.path.exists(db_path):
    os.mkdir(db_path)
db_f_name = os.path.join(db_path, 'test.db')
try:
    os.remove(db_f_name)
except OSError:
    pass
set_engine_and_session('sqlite:///' + db_f_name, echo=False)
