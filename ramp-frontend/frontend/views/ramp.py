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
from rampdb.tools.team import is_user_signed_up

from frontend import db

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
        # get the starting-kit notebook in HTML format
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
