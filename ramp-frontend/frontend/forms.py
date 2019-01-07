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
    user_name = StringField('user_name', [validators.DataRequired()])
    password = PasswordField('password', [validators.DataRequired()])


class UserUpdateProfileForm(FlaskForm):
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
