"""Blueprint for all leaderboard functions for the RAMP frontend."""
import datetime
import logging

import flask_login

from flask import Blueprint
from flask import current_app as app
from flask import redirect
from flask import render_template
from flask import url_for

from ramp_database.tools.event import get_event
from ramp_database.tools.frontend import is_admin
from ramp_database.tools.frontend import is_accessible_code
from ramp_database.tools.frontend import is_accessible_event
from ramp_database.tools.frontend import is_accessible_leaderboard
from ramp_database.tools.frontend import is_user_signed_up
from ramp_database.tools.user import add_user_interaction
from ramp_database.tools.team import get_event_team_by_name

from ramp_frontend import db

from .redirect import redirect_to_user

mod = Blueprint('leaderboard', __name__)
logger = logging.getLogger('RAMP-FRONTEND')

SORTING_COLUMN_INDEX = 2


@mod.route("/events/<event_name>/my_submissions")
@flask_login.login_required
def my_submissions(event_name):
    """Landing page of all user's submission information.

    Parameters
    ----------
    event_name : str
        The name of the event.
    """
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            '{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session, interaction='looking at my_submissions',
            user=flask_login.current_user, event=event
        )
    if not is_accessible_code(db.session, event_name,
                              flask_login.current_user.name):
        error_str = ('No access to my submissions for event {}. If you have '
                     'already signed up, please wait for approval.'
                     .format(event.name))
        return redirect_to_user(error_str)

    # Doesn't work if team mergers are allowed
    event_team = get_event_team_by_name(db.session, event_name,
                                        flask_login.current_user.name)
    leaderboard_html = event_team.leaderboard_html
    failed_leaderboard_html = event_team.failed_leaderboard_html
    new_leaderboard_html = event_team.new_leaderboard_html
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    if event.official_score_type.is_lower_the_better:
        sorting_direction = 'asc'
    else:
        sorting_direction = 'desc'

    return render_template('leaderboard.html',
                           leaderboard_title='Trained submissions',
                           leaderboard=leaderboard_html,
                           failed_leaderboard=failed_leaderboard_html,
                           new_leaderboard=new_leaderboard_html,
                           sorting_column_index=SORTING_COLUMN_INDEX,
                           sorting_direction=sorting_direction,
                           event=event,
                           admin=admin)


@mod.route("/events/<event_name>/leaderboard")
@flask_login.login_required
def leaderboard(event_name):
    """Landing page showing all user's submissions information.

    Parameters
    ----------
    event_name : str
        The name of the event.
    """
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            '{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name))
    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session,
            interaction='looking at leaderboard',
            user=flask_login.current_user,
            event=event
        )

    if is_accessible_leaderboard(db.session, event_name,
                                 flask_login.current_user.name):
        leaderboard_html = event.public_leaderboard_html_with_links
    else:
        leaderboard_html = event.public_leaderboard_html_no_links
    if event.official_score_type.is_lower_the_better:
        sorting_direction = 'asc'
    else:
        sorting_direction = 'desc'

    leaderboard_kwargs = dict(
        leaderboard=leaderboard_html,
        leaderboard_title='Leaderboard',
        sorting_column_index=SORTING_COLUMN_INDEX,
        sorting_direction=sorting_direction,
        event=event
    )

    if is_admin(db.session, event_name, flask_login.current_user.name):
        failed_leaderboard_html = event.failed_leaderboard_html
        new_leaderboard_html = event.new_leaderboard_html
        template = render_template(
            'leaderboard.html',
            failed_leaderboard=failed_leaderboard_html,
            new_leaderboard=new_leaderboard_html,
            admin=True,
            **leaderboard_kwargs
        )
    else:
        template = render_template(
            'leaderboard.html', **leaderboard_kwargs
        )

    return template


@mod.route("/events/<event_name>/competition_leaderboard")
@flask_login.login_required
def competition_leaderboard(event_name):
    """Landing page for the competition leaderboard for all users.

    Parameters
    ----------
    event_name : str
        The event name.
    """
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            '{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session,
            interaction='looking at leaderboard',
            user=flask_login.current_user,
            event=event
        )
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    approved = is_user_signed_up(
        db.session, event_name, flask_login.current_user.name
    )
    asked = approved
    leaderboard_html = event.public_competition_leaderboard_html
    leaderboard_kwargs = dict(
        leaderboard=leaderboard_html,
        leaderboard_title='Leaderboard',
        sorting_column_index=0,
        sorting_direction='asc',
        event=event,
        admin=admin,
        asked=asked,
        approved=approved
    )

    return render_template('leaderboard.html', **leaderboard_kwargs)


@mod.route("/events/<event_name>/private_leaderboard")
@flask_login.login_required
def private_leaderboard(event_name):
    """Landing page for the private leaderboard.

    Parameters
    ----------
    event_name : str
        The event name.
    """
    if not flask_login.current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            '{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if (not is_admin(db.session, event_name, flask_login.current_user.name) and
            (event.closing_timestamp is None or
             event.closing_timestamp > datetime.datetime.utcnow())):
        return redirect(url_for('ramp.problems'))

    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session,
            interaction='looking at private leaderboard',
            user=flask_login.current_user,
            event=event
        )
    leaderboard_html = event.private_leaderboard_html
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    if event.official_score_type.is_lower_the_better:
        sorting_direction = 'asc'
    else:
        sorting_direction = 'desc'

    approved = is_user_signed_up(
        db.session, event_name, flask_login.current_user.name
    )
    asked = approved
    template = render_template(
        'leaderboard.html',
        leaderboard_title='Leaderboard',
        leaderboard=leaderboard_html,
        sorting_column_index=SORTING_COLUMN_INDEX+1,
        sorting_direction=sorting_direction,
        event=event,
        private=True,
        admin=admin,
        asked=asked,
        approved=approved
    )

    return template


@mod.route("/events/<event_name>/private_competition_leaderboard")
@flask_login.login_required
def private_competition_leaderboard(event_name):
    """Landing page for the private competition leaderboard.

    Parameters
    ----------
    event_name : str
        The event name.
    """
    if not flask_login.current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            '{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if (not is_admin(db.session, event_name, flask_login.current_user.name) and
            (event.closing_timestamp is None or
             event.closing_timestamp > datetime.datetime.utcnow())):
        return redirect(url_for('ramp.problems'))

    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session,
            interaction='looking at private leaderboard',
            user=flask_login.current_user,
            event=event
        )

    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    approved = is_user_signed_up(
        db.session, event_name, flask_login.current_user.name
    )
    asked = approved
    leaderboard_html = event.private_competition_leaderboard_html

    leaderboard_kwargs = dict(
        leaderboard=leaderboard_html,
        leaderboard_title='Leaderboard',
        sorting_column_index=0,
        sorting_direction='asc',
        event=event,
        admin=admin,
        asked=asked,
        approved=approved
    )

    return render_template('leaderboard.html', **leaderboard_kwargs)
