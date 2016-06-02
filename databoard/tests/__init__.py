from databoard import app, db

test_app = app.test_client()
app.config.from_object('databoard.config.TestingConfig')


def init_db():
    """Initialisation of a test database"""
    db.session.close()
    db.drop_all()
    db.create_all()
    print(db)


def teardown():
    db.session.remove()
    db.drop_all()


init_db()
