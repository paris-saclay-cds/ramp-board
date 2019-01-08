from flask_wtf import FlaskForm

from wtforms import BooleanField
from wtforms import PasswordField
from wtforms import StringField
from wtforms import validators
from wtforms import ValidationError


def _space_check(form, field):
    if ' ' in field.data:
        raise ValidationError('Field cannot contain space.')


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
