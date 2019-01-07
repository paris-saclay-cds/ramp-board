import flask_login

from flask import Blueprint
from flask import render_template

from rampdb.tools.user import add_user_interaction
from rampdb.tools.event import get_problem

from frontend import db

mod = Blueprint('general', __name__)


@mod.route('/')
def index():
    return render_template('index.html')


@mod.route("/problems")
def problems():
    user = (flask_login.current_user
            if flask_login.current_user.is_authenticated else None)
    add_user_interaction(
            db.session, interaction='looking at problems', user=user
    )

    # problems = Problem.query.order_by(Problem.id.desc())
    problems = get_problem(db.session, None)
    return render_template('problems.html', problems=problems)
