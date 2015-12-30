from flask.ext.wtf import Form
from wtforms import StringField, PasswordField, SelectMultipleField, validators
from wtforms.widgets import ListWidget, CheckboxInput
from flask_wtf.file import FileField


class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


class LoginForm(Form):
    user_name = StringField('user_name', [validators.Required()])
    password = PasswordField('password', [validators.Required()])
    # remember_me = BooleanField('remember_me', default=False)


class CodeForm(Form):
    names_codes = []


class SubmitForm(Form):
    submission_name = StringField('submission_name', [validators.Required()])


class ImportForm(Form):
    selected_f_names = MultiCheckboxField('selected_f_names')


class UploadForm(Form):
    file = FileField('file')
