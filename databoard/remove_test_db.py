from databoard import app, db
app.config.from_object('databoard.config.TestingConfig')


def recreate_test_db():
    """Initialisation of a test database"""
    db.session.close()
    db.drop_all()
    db.create_all()
    print(db)
