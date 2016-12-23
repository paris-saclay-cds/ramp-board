import os
from distutils.util import strtobool
from flask.ext.script import Manager
from flask.ext.migrate import Migrate, MigrateCommand
from databoard import db, app

app.config.from_object('databoard.config.Config')
test_config = os.environ.get('DATABOARD_TEST')
if test_config is not None:
    if strtobool(test_config):
        app.config.from_object('databoard.config.TestingConfig')

migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
