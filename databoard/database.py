from flask_sqlalchemy import SQLAlchemy as SQLAlchemyBase

from rampdb.model.base import set_query_property


class SQLAlchemy(SQLAlchemyBase):
    """Flask extension that integrates alchy with Flask-SQLAlchemy."""

    def __init__(self,
                 app=None,
                 use_native_unicode=True,
                 session_options=None,
                 Model=None):
        self.Model = Model

        super(SQLAlchemy, self).__init__(app,
                                         use_native_unicode,
                                         session_options)

    def make_declarative_base(self):
        """Creates or extends the declarative base."""
        if self.Model is None:
            self.Model = super(SQLAlchemyBase, self).make_declarative_base()
        else:
            set_query_property(self.Model, self.session)
        return self.Model
