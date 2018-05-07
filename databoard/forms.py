from __future__ import print_function

from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import (
    StringField, PasswordField, SelectMultipleField, BooleanField,
    IntegerField, DateField, DateTimeField, validators)
from wtforms.widgets import ListWidget, CheckboxInput
from wtforms.validators import ValidationError


def ascii_check(form, field):
    try:
        # XXX may not work in python 3, should probably be field.data.encode
        field.data.decode('ascii')
    except Exception:
        print('bla')
        raise ValidationError('Field cannot contain non-ascii characters.')


def space_check(form, field):
    if ' ' in field.data:
        raise ValidationError('Field cannot contain space.')


class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


class LoginForm(FlaskForm):
    user_name = StringField('user_name', [validators.Required()])
    password = PasswordField('password', [validators.Required()])


class UserUpdateProfileForm(FlaskForm):
    user_name = StringField('user_name', [
        validators.Required(), validators.Length(min=1, max=20), space_check])
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


class EventUpdateProfileForm(FlaskForm):
    suffix = StringField('event_suffix', [
        validators.Length(max=20), ascii_check, space_check])
    title = StringField('event_title', [
        validators.Required(), validators.Length(max=80)])
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


class AskForEventForm(FlaskForm):
    suffix = StringField('event_suffix', [
        validators.Required(), validators.Length(max=20), ascii_check,
        space_check])
    title = StringField('event_title', [
        validators.Required(), validators.Length(max=80)])
    n_students = IntegerField('n_students', [
        validators.Required(), validators.NumberRange(min=0)])
    min_duration_between_submissions_hour = IntegerField('min_h', [
        validators.NumberRange(min=0)])
    min_duration_between_submissions_minute = IntegerField('min_m', [
        validators.NumberRange(min=0, max=59)])
    min_duration_between_submissions_second = IntegerField('min_s', [
        validators.NumberRange(min=0, max=59)])
    opening_date = DateField('opening_date', [
        validators.Required()], format='%Y-%m-%d')
    closing_date = DateField('closing_date', [
        validators.Required()], format='%Y-%m-%d')


class CodeForm(FlaskForm):
    names_codes = []


class SubmitForm(FlaskForm):
    submission_name = StringField('submission_name', [
        validators.Required(), space_check])


class CreditForm(FlaskForm):
    note = StringField('submission_name')
    self_credit = StringField('self credit')
    name_credits = []


class ImportForm(FlaskForm):
    selected_f_names = MultiCheckboxField('selected_f_names')


class UploadForm(FlaskForm):
    file = FileField('file')


class EmailForm(FlaskForm):
    email = StringField('Email', validators=[
        validators.DataRequired(), validators.Email()])


class PasswordForm(FlaskForm):
    password = PasswordField(
        'Password', validators=[validators.DataRequired()])
