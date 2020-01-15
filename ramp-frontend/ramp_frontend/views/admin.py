"""Blueprint for all admin functions for the RAMP frontend."""
import logging

import flask_login

from flask import Blueprint
from flask import flash
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for

from sqlalchemy.exc import IntegrityError

from ramp_database.model import Event
from ramp_database.model import EventTeam
from ramp_database.model import Submission
from ramp_database.model import User

from ramp_database.exceptions import NameClashError

from ramp_database.tools.event import get_event
from ramp_database.tools.frontend import is_admin
from ramp_database.tools.frontend import is_accessible_event
from ramp_database.tools.frontend import is_user_signed_up
from ramp_database.tools.frontend import is_user_sign_up_requested
from ramp_database.tools.user import approve_user
from ramp_database.tools.user import delete_user
from ramp_database.tools.user import select_user_by_name
from ramp_database.tools.user import get_user_interactions_by_name
from ramp_database.tools.team import delete_event_team
from ramp_database.tools.team import sign_up_team

from ramp_frontend import db

from ..forms import EventUpdateProfileForm
from ..utils import send_mail

from .redirect import redirect_to_user

mod = Blueprint('admin', __name__)
logger = logging.getLogger('RAMP-FRONTEND')


@mod.route("/approve_users", methods=['GET', 'POST'])
@flask_login.login_required
def approve_users():
    """Approve new user to log-in and sign-up to events."""
    if not flask_login.current_user.access_level == 'admin':
        return redirect_to_user(
            'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    if request.method == 'GET':
        # TODO: replace by some get_functions
        asked_users = User.query.filter_by(access_level='asked').all()
        asked_sign_up = EventTeam.query.filter_by(approved=False).all()
        return render_template('approve.html',
                               asked_users=asked_users,
                               asked_sign_up=asked_sign_up,
                               admin=True)
    elif request.method == 'POST':
        users_to_be_approved = request.form.getlist('approve_users')
        event_teams_to_be_approved = request.form.getlist(
            'approve_event_teams'
        )
        message = "{}d users:\n".format(request.form["submit_button"][:-1])
        for asked_user in users_to_be_approved:
            user = select_user_by_name(db.session, asked_user)
            if request.form["submit_button"] == "Approve!":
                approve_user(db.session, asked_user)

                subject = 'Your RAMP account has been approved'
                body = ('{}, your account has been approved. You can now '
                        'sign-up for any open RAMP event.'
                        .format(user.name))
                send_mail(
                    to=user.email, subject=subject, body=body
                )
            elif request.form["submit_button"] == "Remove!":
                delete_user(db.session, asked_user)
            message += "{}\n".format(asked_user)

        message += "{}d event_team:\n".format(
            request.form["submit_button"][:-1]
        )
        for asked_id in event_teams_to_be_approved:
            asked_event_team = EventTeam.query.get(asked_id)
            user = select_user_by_name(db.session, asked_event_team.team.name)

            if request.form["submit_button"] == "Approve!":
                sign_up_team(db.session, asked_event_team.event.name,
                             asked_event_team.team.name)

                subject = ('Signed up for the RAMP event {}'
                           .format(asked_event_team.event.name))
                body = ('{}, you have been registered to the RAMP event {}. '
                        'You can now proceed to your sandbox and make '
                        'submissions.\nHave fun!!!'
                        .format(user.name, asked_event_team.event.name))
                send_mail(
                    to=user.email, subject=subject, body=body
                )
            elif request.form["submit_button"] == "Remove!":
                delete_event_team(
                    db.session, asked_event_team.event.name,
                    asked_event_team.team.name
                )
            message += "{}\n".format(asked_event_team)
        return redirect_to_user(
            message, is_error=False,
            category="{}d users".format(request.form["submit_button"][:-1])
        )


@mod.route("/sign_up/<user_name>")
@flask_login.login_required
def approve_single_user(user_name):
    """Approve a single user. This is usually used to approve user through
    email."""
    if not flask_login.current_user.access_level == 'admin':
        return redirect_to_user(
            'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    user = User.query.filter_by(name=user_name).one_or_none()
    if not user:
        return redirect_to_user(
            'No user {}'.format(user_name), is_error=True
        )
    approve_user(db.session, user.name)
    return redirect_to_user(
        '{} is signed up'.format(user), is_error=False,
        category='Successful sign-up'
    )


@mod.route("/events/<event_name>/sign_up/<user_name>")
@flask_login.login_required
def approve_sign_up_for_event(event_name, user_name):
    """Approve a user for a specific event.

    This way of approval is usually used by clicking in an email sent to the
    admin.

    Parameters
    ----------
    event_name : str
        The name of the event.
    user_name : str
        The name of the user.
    """
    event = get_event(db.session, event_name)
    user = User.query.filter_by(name=user_name).one_or_none()
    if not is_admin(db.session, event_name, flask_login.current_user.name):
        return redirect_to_user('Sorry {}, you do not have admin rights'
                                .format(flask_login.current_user.firstname),
                                is_error=True)
    if not event or not user:
        return redirect_to_user('No event {} or no user {}'
                                .format(event_name, user_name), is_error=True)
    sign_up_team(db.session, event.name, user.name)

    subject = ('Signed up for the RAMP event {}'
               .format(event.name))
    body = ('{}, you have been registered to the RAMP event {}. '
            'You can now proceed to your sandbox and make submissions.'
            '\nHave fun!!!'.format(user.name, event.name))
    send_mail(to=user.email, subject=subject, body=body)

    return redirect_to_user('{} is signed up for {}.'.format(user, event),
                            is_error=False, category='Successful sign-up')


@mod.route("/events/<event_name>/update", methods=['GET', 'POST'])
@flask_login.login_required
def update_event(event_name):
    """Update the parameters of an event.

    Parameters
    ----------
    event_name : str
        The name of the event.
    """
    if not is_admin(db.session, event_name, flask_login.current_user.name):
        return redirect_to_user(
            'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            '{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    logger.info('{} is updating event {}'
                .format(flask_login.current_user.name, event.name))
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    # We assume here that event name has the syntax <problem_name>_<suffix>

    h = event.min_duration_between_submissions // 3600
    m = event.min_duration_between_submissions // 60 % 60
    s = event.min_duration_between_submissions % 60
    form = EventUpdateProfileForm(
        title=event.title,
        is_send_trained_mails=event.is_send_trained_mails,
        is_send_submitted_mails=event.is_send_submitted_mails,
        is_public=event.is_public,
        is_controled_signup=event.is_controled_signup,
        is_competitive=event.is_competitive,
        min_duration_between_submissions_hour=h,
        min_duration_between_submissions_minute=m,
        min_duration_between_submissions_second=s,
        opening_timestamp=event.opening_timestamp,
        closing_timestamp=event.closing_timestamp,
        public_opening_timestamp=event.public_opening_timestamp,
    )
    if form.validate_on_submit():
        try:
            event.title = form.title.data
            event.is_send_trained_mails = form.is_send_trained_mails.data
            event.is_send_submitted_mails = form.is_send_submitted_mails.data
            event.is_public = form.is_public.data
            event.is_controled_signup = form.is_controled_signup.data
            event.is_competitive = form.is_competitive.data
            event.min_duration_between_submissions = (
                form.min_duration_between_submissions_hour.data * 3600 +
                form.min_duration_between_submissions_minute.data * 60 +
                form.min_duration_between_submissions_second.data)
            event.opening_timestamp = form.opening_timestamp.data
            event.closing_timestamp = form.closing_timestamp.data
            event.public_opening_timestamp = form.public_opening_timestamp.data
            db.session.commit()

        except IntegrityError as e:
            db.session.rollback()
            message = ''
            existing_event = get_event(db.session, event.name)
            if existing_event is not None:
                message += 'event name is already in use'
            # # try:
            # #     User.query.filter_by(email=email).one()
            # #     if len(message) > 0:
            # #         message += ' and '
            # #     message += 'email is already in use'
            # except NoResultFound:
            #     pass
            if message:
                e = NameClashError(message)
            flash('{}'.format(e), category='Update event error')
            return redirect(url_for('update_event', event_name=event.name))

        return redirect(url_for('ramp.problems'))

    approved = is_user_signed_up(
        db.session, event_name, flask_login.current_user.name
    )
    asked = is_user_sign_up_requested(
        db.session, event_name, flask_login.current_user.name
    )
    return render_template(
        'update_event.html',
        form=form,
        event=event,
        admin=admin,
        asked=asked,
        approved=approved
    )


@mod.route("/user_interactions")
@flask_login.login_required
def user_interactions():
    """Show the user interactions recorded on the website."""
    if flask_login.current_user.access_level != 'admin':
        return redirect_to_user(
            'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    user_interactions_html = get_user_interactions_by_name(
        db.session, output_format='html'
    )
    return render_template(
        'user_interactions.html',
        user_interactions_title='User interactions',
        user_interactions=user_interactions_html
    )


@mod.route("/events/<event_name>/dashboard_submissions")
@flask_login.login_required
def dashboard_submissions(event_name):
    """Show information about all submissions for a given event.

    Parameters
    ----------
    event_name : str
        The name of the event.
    """
    if not is_admin(db.session, event_name, flask_login.current_user.name):
        return redirect_to_user(
            'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    event = get_event(db.session, event_name)
    # Get dates and number of submissions
    submissions = \
        (Submission.query
                   .filter(Event.name == event.name)
                   .filter(Event.id == EventTeam.event_id)
                   .filter(EventTeam.id == Submission.event_team_id)
                   .order_by(Submission.submission_timestamp)
                   .all())
    submissions = [sub for sub in submissions if sub.is_not_sandbox]
    timestamp_submissions = [
        sub.submission_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        for sub in submissions]
    name_submissions = [sub.name for sub in submissions]
    cumulated_submissions = list(range(1, 1 + len(submissions)))
    training_sec = [
        (
            sub.training_timestamp - sub.submission_timestamp
        ).total_seconds() / 60.
        if sub.training_timestamp is not None else 0
        for sub in submissions
    ]
    dashboard_kwargs = {'event': event,
                        'timestamp_submissions': timestamp_submissions,
                        'training_sec': training_sec,
                        'cumulated_submissions': cumulated_submissions,
                        'name_submissions': name_submissions}
    failed_leaderboard_html = event.failed_leaderboard_html
    new_leaderboard_html = event.new_leaderboard_html
    approved = is_user_signed_up(
        db.session, event_name, flask_login.current_user.name
    )
    asked = is_user_sign_up_requested(
        db.session, event_name, flask_login.current_user.name
    )
    return render_template(
        'dashboard_submissions.html',
        failed_leaderboard=failed_leaderboard_html,
        new_leaderboard=new_leaderboard_html,
        admin=True,
        approved=approved,
        asked=asked,
        **dashboard_kwargs)
