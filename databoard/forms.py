from flask.ext.wtf import Form
from wtforms import StringField, BooleanField, PasswordField, validators


class LoginForm(Form):
    user_name = StringField(
        'user_name', [validators.Required()])
    password = PasswordField(
        'password', [validators.Required()])
    # remember_me = BooleanField('remember_me', default=False)
