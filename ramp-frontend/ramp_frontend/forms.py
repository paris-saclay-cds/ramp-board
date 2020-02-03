"""
The :mod:`ramp_frontend.forms` module defines the different forms used on the
website.
"""

from flask_wtf import FlaskForm

from wtforms import BooleanField
from wtforms import DateField
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
        field.data.encode('ascii')
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
    """Code form.

    This is the form used to contain the code when submitting to RAMP.

    Attributes
    ----------
    named_codes : list of tuple (submission_file_name, submission_code)
        The place holder containing the name of the submission file and the
        code associated.
    """
    names_codes = []


class SubmitForm(FlaskForm):
    """Submission name form.

    This is the form where the name of the submission given by the user will
    be stored.

    Attributes
    ----------
    submission_name : str
        The name of the submission.
    """
    submission_name = StringField('submission_name',
                                  [validators.DataRequired(), _space_check])


class UploadForm(FlaskForm):
    """Submission uploading form.

    This is the form used to upload a file to be loaded during a RAMP
    submission.

    Attributes
    ----------
    file : file
        File to be uploaded and loaded into the sandbox code form.
    """
    file = FileField('file')


class EventUpdateProfileForm(FlaskForm):
    """Form to update the parameters of an event.

    Attributes
    ----------
    title : str
        The event title.
    is_send_trained_mails : bool
        Whether or not to send an email when submissions are trained.
    is_public : bool
        Whether or not the event is public.
    is_controled_signup : bool
        Whether or not the event has controlled sign-up.
    is_competitive : bool
        Whether or not the event has a competitive phase.
    min_duration_between_submission_hour : int
        The number of hour to wait between two submissions.
    min_duration_between_submission_minute : int
        The number of minute to wait between two submissions.
    min_duration_between_submission_second : int
        The number of second to wait between two submissions.
    opening_timestamp : datetime
        The date and time when the event is opening.
    closing_timestamp : datetime
        The date and time when the event is closing.
    public_opening_timestamp : datetime
        The date and time when the public phase of the event is opening.
    """
    title = StringField(
        'event_title', [validators.DataRequired(), validators.Length(max=80)]
    )
    is_send_trained_mails = BooleanField()
    is_send_submitted_mails = BooleanField()
    is_public = BooleanField()
    is_controled_signup = BooleanField()
    is_competitive = BooleanField()
    min_duration_between_submissions_hour = IntegerField(
        'min_h', [validators.NumberRange(min=0)]
    )
    min_duration_between_submissions_minute = IntegerField(
        'min_m', [validators.NumberRange(min=0, max=59)]
    )
    min_duration_between_submissions_second = IntegerField(
        'min_s', [validators.NumberRange(min=0, max=59)]
    )
    opening_timestamp = DateTimeField(
        'opening_timestamp', [], format='%Y-%m-%d %H:%M:%S'
    )
    closing_timestamp = DateTimeField(
        'closing_timestamp', [], format='%Y-%m-%d %H:%M:%S'
    )
    public_opening_timestamp = DateTimeField(
        'public_opening_timestamp', [], format='%Y-%m-%d %H:%M:%S'
    )


class MultiCheckboxField(SelectMultipleField):
    """A form containing multiple checkboxes."""
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


class ImportForm(FlaskForm):
    """The form allowing to select which model to view."""
    selected_f_names = MultiCheckboxField('selected_f_names')


class CreditForm(FlaskForm):
    """Credit form.

    The credit form is used to acknowledge other submission when making a
    RAMP submission after tracking the user activity.

    Attributes
    ----------
    note : str
        Some notes regarding the credit.
    self_credit : str
        The credit given to the current submission.
    name_credits : list
        The name for the credits.
    """
    note = StringField('submission_name')
    self_credit = StringField('self credit')
    name_credits = []


class AskForEventForm(FlaskForm):
    """Form to ask for a new event.

    Attributes
    ----------
    suffix : str
        The suffix used for the event.
    title : str
        The event title.
    n_students : int
        The number of students that will take part in the event.
    min_duration_between_submission_hour : int
        The number of hour to wait between two submissions.
    min_duration_between_submission_minute : int
        The number of minute to wait between two submissions.
    min_duration_between_submission_second : int
        The number of second to wait between two submissions.
    opening_timestamp : datetime
        The date and time when the event is opening.
    closing_timestamp : datetime
        The date and time when the event is closing.
    """
    suffix = StringField(
        'event_suffix',
        [validators.DataRequired(), validators.Length(max=20), _ascii_check,
         _space_check]
    )
    title = StringField(
        'event_title',
        [validators.DataRequired(), validators.Length(max=80)]
    )
    n_students = IntegerField(
        'n_students',
        [validators.DataRequired(), validators.NumberRange(min=0)]
    )
    min_duration_between_submissions_hour = IntegerField(
        'min_h', [validators.NumberRange(min=0)]
    )
    min_duration_between_submissions_minute = IntegerField(
        'min_m', [validators.NumberRange(min=0, max=59)]
    )
    min_duration_between_submissions_second = IntegerField(
        'min_s', [validators.NumberRange(min=0, max=59)]
    )
    opening_date = DateField(
        'opening_date', [validators.DataRequired()], format='%Y-%m-%d'
    )
    closing_date = DateField(
        'closing_date', [validators.DataRequired()], format='%Y-%m-%d'
    )


class EmailForm(FlaskForm):
    email = StringField(
        'Email', validators=[validators.DataRequired(), validators.Email()]
    )


class PasswordForm(FlaskForm):
    password = PasswordField(
        'Password', validators=[validators.DataRequired()]
    )
