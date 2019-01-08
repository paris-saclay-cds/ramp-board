import codecs
import logging
import os

import flask_login

from flask import Blueprint
from flask import render_template

from rampdb.tools.event import get_event
from rampdb.tools.event import get_problem
from rampdb.tools.event import is_accessible_event
from rampdb.tools.event import is_admin
from rampdb.tools.user import add_user_interaction
from rampdb.tools.user import get_user_by_name
from rampdb.tools.team import ask_sign_up_team
from rampdb.tools.team import sign_up_team
from rampdb.tools.team import is_user_signed_up

from frontend import db

from .redirect import redirect_to_sandbox
from .redirect import redirect_to_user

mod = Blueprint('ramp', __name__)
logger = logging.getLogger('FRONTEND')


@mod.route("/problems")
def problems():
    """Problems request."""
    user = (flask_login.current_user
            if flask_login.current_user.is_authenticated else None)
    add_user_interaction(
        db.session, interaction='looking at problems', user=user
    )

    # problems = Problem.query.order_by(Problem.id.desc())
    return render_template('problems.html',
                           problems=get_problem(db.session, None))


@mod.route("/problems/<problem_name>")
def problem(problem_name):
    problem = get_problem(db.session, problem_name)
    if problem:
        if flask_login.current_user.is_authenticated:
            add_user_interaction(
                db.session,
                interaction='looking at problem',
                user=flask_login.current_user,
                problem=problem
            )
        else:
            add_user_interaction(
                db.session, interaction='looking at problem', problem=problem
            )
        description_f_name = os.path.join(
            problem.path_ramp_kits, problem.name,
            '{}_starting_kit.html'.format(problem.name)
        )
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        return render_template('problem.html', problem=problem,
                               description=description)
    else:
        return redirect_to_user(u'Problem {} does not exist.'
                                .format(problem_name), is_error=True)


@mod.route("/events/<event_name>")
@flask_login.login_required
def user_event(event_name):
    if flask_login.current_user.access_level == 'asked':
        msg = 'Your account has not been approved yet by the administrator'
        logger.error(msg)
        return redirect_to_user(msg)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(u'{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    event = get_event(db.session, event_name)
    if event:
        if flask_login.current_user.is_authenticated:
            add_user_interaction(db.session, interaction='looking at event',
                                 user=flask_login.current_user, event=event)
        else:
            add_user_interaction(db.session, interaction='looking at event',
                                 event=event)
        description_f_name = os.path.join(
            event.problem.path_ramp_kits,
            event.problem.name,
            '{}_starting_kit.html'.format(event.problem.name)
        )
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        admin = is_admin(db.session, event_name, flask_login.current_user.name)
        if flask_login.current_user.is_anonymous:
            approved = False
            asked = False
        else:
            approved = is_user_signed_up(
                db.session, event_name, flask_login.current_user.name
            )
            asked = approved
        return render_template('event.html',
                               description=description,
                               event=event,
                               admin=admin,
                               approved=approved,
                               asked=asked)
    return redirect_to_user(u'Event {} does not exist.'
                            .format(event_name), is_error=True)


@mod.route("/events/<event_name>/sign_up/<user_name>")
@flask_login.login_required
def approve_sign_up_for_event(event_name, user_name):
    event = get_event(db.session, event_name)
    user = get_user_by_name(db.session, user_name)
    if not is_admin(db.session, event_name, flask_login.current_user.name):
        return redirect_to_user(u'Sorry {}, you do not have admin rights'
                                .format(flask_login.current_user.firstname),
                                is_error=True)
    if not event or not user:
        return redirect_to_user(u'Oups, no event {} or no user {}.'
                                .format(event_name, user_name), is_error=True)
    sign_up_team(db.session, event.name, user.name)
    return redirect_to_user(u'{} is signed up for {}.'.format(user, event),
                            is_error=False, category='Successful sign-up')


@mod.route("/events/<event_name>/sign_up")
@flask_login.login_required
def sign_up_for_event(event_name):
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(u'{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    add_user_interaction(db.session, interaction='signing up at event',
                         user=flask_login.current_user, event=event)

    ask_sign_up_team(db.session, event.name, flask_login.current_user.name)
    if event.is_controled_signup:
        # send_sign_up_request_mail(event, flask_login.current_user)
        return redirect_to_user("Sign-up request is sent to event admins.",
                                is_error=False, category='Request sent')
    else:
        sign_up_team(event.name, flask_login.current_user.name)
        return redirect_to_sandbox(
            event,
            u'{} is signed up for {}.'
            .format(flask_login.current_user.firstname, event),
            is_error=False,
            category='Successful sign-up'
        )
