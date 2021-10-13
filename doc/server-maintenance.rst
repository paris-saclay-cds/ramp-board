##################
Server maintenance
##################

Database migrations
-------------------

Please follow the
`flask-migrate <https://flask-migrate.readthedocs.io/en/latest/>`_ documentation
to perform database migrations.

Note that the corresponding migration files **should not** be added to the
ramp-board repository but rather to a separate per-instance git repository.

You may need to add the following file to this repository,

.. code:: python

    from flask_sqlalchemy import SQLAlchemy
    from flask_migrate import Migrate

    from ramp_frontend.wsgi import make_app
    from ramp_database.model import Model

    app = make_app("<path to the config.yaml>")

    db = SQLAlchemy(model_class=Model)
    Migrate(app, db)


    if __name__ == "__main__":
        app.run()

then run the appropriate `flask db` commands from within it.
