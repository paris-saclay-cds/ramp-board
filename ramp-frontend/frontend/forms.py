from flask_wtf import FlaskForm

import six

from wtforms import BooleanField
from wtforms import DateTimeField
from wtforms import FileField
from wtforms import IntegerField
from wtforms import PasswordField
from wtforms import SelectMultipleField
from wtforms import StringField
from wtforms import validators
from wtforms import ValidationError
from wtforms.widgets import CheckboxInput
from wtforms.widgets import ListWidget


def _space_check(form, field):
    if ' ' in field.data:
        raise ValidationError('Field cannot contain space.')


def _ascii_check(form, field):
    try:
        # XXX may not work in python 3, should probably be field.data.encode
        if six.PY3:
            field.data.encode('ascii')
        else:
            field.data.decode('ascii')
    except Exception:
        raise ValidationError('Field cannot contain non-ascii characters.')


class LoginForm(FlaskForm):
    """Login-in form.

    Attributes
    ----------
    user_name : str
        The user name.
    password : str
        The user password.
    """
    user_name = StringField('user_name', [validators.DataRequired()])
    password = PasswordField('password', [validators.DataRequired()])


class UserUpdateProfileForm(FlaskForm):
    """User profile form.

    Attributes
    ----------
    user_name : str
        The user name.
    firstname : str
        The user's first name.
    lastname : str
        The user's last name.
    email : str
        The user's email address.
    linkedin_url : str, default == ''
        The user's LinkedIn URL.
    twitter_url : str, defaut == ''
        The user's Twitter URL.
    facebook_url : str, default == ''
        The user's Facebook URL.
    google_url : str, default == ''
        The user's Google URL.
    github_url : str, default == ''
        The user's GitHub URL.
    website_url : str, default == ''
        The user's website URL.
    bio : str, default == ''
        The user's bio.
    is_want_news : bool, default is True
        Whether the user want some info from us.
    """
    user_name = StringField('user_name', [
        validators.DataRequired(), validators.Length(min=1, max=20),
        _space_check])
    firstname = StringField('firstname', [validators.DataRequired()])
    lastname = StringField('lastname', [validators.DataRequired()])
    email = StringField('email', [validators.DataRequired()])
    linkedin_url = StringField('linkedin_url')
    twitter_url = StringField('twitter_url')
    facebook_url = StringField('facebook_url')
    google_url = StringField('google_url')
    github_url = StringField('github_url')
    website_url = StringField('website_url')
    bio = StringField('bio')
    is_want_news = BooleanField()


class UserCreateProfileForm(UserUpdateProfileForm):
    """User profile form.

    Attributes
    ----------
    user_name : str
        The user name.
    password : str
        The user password.
    firstname : str
        The user's first name.
    lastname : str
        The user's last name.
    email : str
        The user's email address.
    linkedin_url : str, default == ''
        The user's LinkedIn URL.
    twitter_url : str, defaut == ''
        The user's Twitter URL.
    facebook_url : str, default == ''
        The user's Facebook URL.
    google_url : str, default == ''
        The user's Google URL.
    github_url : str, default == ''
        The user's GitHub URL.
    website_url : str, default == ''
        The user's website URL.
    bio : str, default == ''
        The user's bio.
    is_want_news : bool, default is True
        Whether the user want some info from us.
    """
    password = PasswordField('password', [validators.DataRequired()])


class CodeForm(FlaskForm):
    names_codes = []


class SubmitForm(FlaskForm):
    submission_name = StringField('submission_name',
                                  [validators.DataRequired(), _space_check])


class UploadForm(FlaskForm):
    file = FileField('file')


class EventUpdateProfileForm(FlaskForm):
    suffix = StringField('event_suffix', [
        validators.Length(max=20), _ascii_check, _space_check])
    title = StringField('event_title', [
        validators.DataRequired(), validators.Length(max=80)])
    is_send_trained_mails = BooleanField()
    is_send_submitted_mails = BooleanField()
    is_public = BooleanField()
    is_controled_signup = BooleanField()
    is_competitive = BooleanField()
    min_duration_between_submissions_hour = IntegerField('min_h', [
        validators.NumberRange(min=0)])
    min_duration_between_submissions_minute = IntegerField('min_m', [
        validators.NumberRange(min=0, max=59)])
    min_duration_between_submissions_second = IntegerField('min_s', [
        validators.NumberRange(min=0, max=59)])
    opening_timestamp = DateTimeField(
        'opening_timestamp', [], format='%Y-%m-%d %H:%M:%S')
    closing_timestamp = DateTimeField(
        'closing_timestamp', [], format='%Y-%m-%d %H:%M:%S')
    public_opening_timestamp = DateTimeField(
        'public_opening_timestamp', [], format='%Y-%m-%d %H:%M:%S')


class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


class ImportForm(FlaskForm):
    selected_f_names = MultiCheckboxField('selected_f_names')
