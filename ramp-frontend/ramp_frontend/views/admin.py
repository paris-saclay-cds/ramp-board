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
from ramp_database.tools.user import approve_user
from ramp_database.tools.user import select_user_by_name
from ramp_database.tools.user import get_user_interactions_by_name
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
            u'Sorry {}, you do not have admin rights'
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
        message = "Approved users:\n"
        for asked_user in users_to_be_approved:
            approve_user(db.session, asked_user)
            user = select_user_by_name(db.session, asked_user)

            subject = 'Your RAMP account have been approved'
            body = ('{}, your account have been approved. You can now sign-up '
                    'for any opened RAMP event.'
                    .format(user.name))
            send_mail(
                to=user.email, subject=subject, body=body
            )

            message += "{}\n".format(asked_user)
        message += "Approved event_team:\n"
        for asked_id in event_teams_to_be_approved:
            asked_event_team = EventTeam.query.get(asked_id)
            sign_up_team(db.session, asked_event_team.event.name,
                         asked_event_team.team.name)

            subject = ('Signed up for the RAMP event {}'
                       .format(asked_event_team.event.name))
            body = ('{}, you have been registered to the RAMP event {}. '
                    'You can now proceed to your sandbox and make submission.'
                    '\nHave fun!!!'.format(flask_login.current_user.name,
                                           asked_event_team.event.name))
            send_mail(
                to=flask_login.current_user.email, subject=subject, body=body
            )
            message += "{}\n".format(asked_event_team)
        return redirect_to_user(message, is_error=False,
                                category="Approved users")


@mod.route("/sign_up/<user_name>")
@flask_login.login_required
def approve_single_user(user_name):
    """Approve a single user. This is usually used to approve user through
    email."""
    if not flask_login.current_user.access_level == 'admin':
        return redirect_to_user(
            u'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    user = User.query.filter_by(name=user_name).one_or_none()
    if not user:
        return redirect_to_user(
            u'No user {}'.format(user_name), is_error=True
        )
    approve_user(db.session, user.name)
    return redirect_to_user(
        u'{} is signed up'.format(user), is_error=False,
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
        return redirect_to_user(u'Sorry {}, you do not have admin rights'
                                .format(flask_login.current_user.firstname),
                                is_error=True)
    if not event or not user:
        return redirect_to_user(u'No event {} or no user {}'
                                .format(event_name, user_name), is_error=True)
    sign_up_team(db.session, event.name, user.name)

    subject = ('Signed up for the RAMP event {}'
               .format(event.name))
    body = ('{}, you have been registered to the RAMP event {}. '
            'You can now proceed to your sandbox and make submission.'
            '\nHave fun!!!'.format(flask_login.current_user.name,
                                   event.name))
    send_mail(
        to=flask_login.current_user.email, subject=subject, body=body
    )

    return redirect_to_user(u'{} is signed up for {}.'.format(user, event),
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
            u'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            u'{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    logger.info(u'{} is updating event {}'
                .format(flask_login.current_user.name, event.name))
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    # We assume here that event name has the syntax <problem_name>_<suffix>
    suffix = event.name[len(event.problem.name) + 1:]

    h = event.min_duration_between_submissions // 3600
    m = event.min_duration_between_submissions // 60 % 60
    s = event.min_duration_between_submissions % 60
    form = EventUpdateProfileForm(
        suffix=suffix, title=event.title,
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
            if form.suffix.data == '':
                event.name = event.problem.name
            else:
                event.name = event.problem.name + '_' + form.suffix.data
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
            flash(u'{}'.format(e), category='Update event error')
            return redirect(url_for('update_event', event_name=event.name))

        return redirect(url_for('ramp.problems'))

    return render_template(
        'update_event.html',
        form=form,
        event=event,
        admin=admin,
    )


@mod.route("/user_interactions")
@flask_login.login_required
def user_interactions():
    """Show the user interactions recorded on the website."""
    if flask_login.current_user.access_level != 'admin':
        return redirect_to_user(
            u'Sorry {}, you do not have admin rights'
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
            u'Sorry {}, you do not have admin rights'
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
    cumulated_submissions = range(1, 1 + len(submissions))
    training_sec = [
        (sub.training_timestamp - sub.submission_timestamp).seconds / 60.
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
    return render_template(
        'dashboard_submissions.html',
        failed_leaderboard=failed_leaderboard_html,
        new_leaderboard=new_leaderboard_html,
        admin=True,
        **dashboard_kwargs)
