from flask.ext.script import Manager
from flask.ext.migrate import Migrate, MigrateCommand
from databoard import db, app

app.config.from_object('databoard.config.Config')
if os.environ.get('DATABOARD_TEST'):
    app.config.from_object('databoard.config.TestingConfig')

migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
