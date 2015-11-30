from flask.ext.wtf import Form
from wtforms import StringField, PasswordField, validators
from wtforms.widgets import TextArea


class LoginForm(Form):
    user_name = StringField('user_name', [validators.Required()])
    password = PasswordField('password', [validators.Required()])
    # remember_me = BooleanField('remember_me', default=False)


class CodeForm(Form):
    code = StringField(u'Text', widget=TextArea())


class SubmitForm(Form):
    submission_name = StringField('submission_name', [validators.Required()])
