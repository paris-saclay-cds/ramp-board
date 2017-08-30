from flask_wtf import Form
from flask_wtf.file import FileField
from wtforms import (
    StringField, PasswordField, SelectMultipleField, BooleanField, validators)
from wtforms.widgets import ListWidget, CheckboxInput


class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


class LoginForm(Form):
    user_name = StringField('user_name', [validators.Required()])
    password = PasswordField('password', [validators.Required()])


class UserUpdateProfileForm(Form):
    user_name = StringField('user_name', [
        validators.Required(), validators.Length(min=1, max=20)])
    firstname = StringField('firstname', [validators.Required()])
    lastname = StringField('lastname', [validators.Required()])
    email = StringField('email', [validators.Required()])
    linkedin_url = StringField('linkedin_url')
    twitter_url = StringField('twitter_url')
    facebook_url = StringField('facebook_url')
    google_url = StringField('google_url')
    github_url = StringField('github_url')
    website_url = StringField('website_url')
    bio = StringField('bio')
    is_want_news = BooleanField()


class UserCreateProfileForm(UserUpdateProfileForm):
    password = PasswordField('password', [validators.Required()])


class CodeForm(Form):
    names_codes = []


class SubmitForm(Form):
    submission_name = StringField('submission_name', [validators.Required()])


class CreditForm(Form):
    note = StringField('submission_name')
    self_credit = StringField('self credit')
    name_credits = []


class ImportForm(Form):
    selected_f_names = MultiCheckboxField('selected_f_names')


class UploadForm(Form):
    file = FileField('file')


class EmailForm(Form):
    email = StringField('Email', validators=[
        validators.DataRequired(), validators.Email()])


class PasswordForm(Form):
    password = PasswordField(
        'Password', validators=[validators.DataRequired()])
