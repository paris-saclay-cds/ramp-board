from flask import Blueprint
from flask import render_template
import flask_login
import os as os

from ramp_database.model import Keyword
from ramp_database.model import Problem

from .redirect import redirect_to_user
from .._version import __version__

mod = Blueprint('general', __name__)


@mod.route('/')
def index():
    """Default landing page."""
    img_ext = ('.png', '.jpg', '.jpeg', '.gif', '.svg')
    current_dir = os.path.dirname(__file__)
    img_folder = os.path.join(current_dir, "..", "static", "img", "powered_by")
    context = {}
    if os.path.isdir(img_folder):
        images = [f for f in os.listdir(img_folder)
                  if f.endswith(img_ext)]
        context["images"] = images
    context["version"] = __version__
    return render_template('index.html', **context)


@mod.route("/description")
def ramp():
    """RAMP description request."""
    user = (flask_login.current_user
            if flask_login.current_user.is_authenticated else None)
    admin = user.access_level == 'admin' if user is not None else False
    return render_template('ramp_description.html', admin=admin)


@mod.route("/data_domains")
def data_domains():
    """Review of all possible keyword attached to the different RAMP
    problems."""
    current_keywords = Keyword.query.order_by(Keyword.name)
    current_problems = Problem.query.order_by(Problem.id)
    return render_template('data_domains.html',
                           keywords=current_keywords,
                           problems=current_problems)


@mod.route("/teaching")
def teaching():
    """Page related to RAMP offers for teaching classes."""
    return render_template('teaching.html')


@mod.route("/data_science_themes")
def data_science_themes():
    """Page reviewing problems organized by ML themes."""
    current_keywords = Keyword.query.order_by(Keyword.name)
    return render_template('data_science_themes.html',
                           keywords=current_keywords)


@mod.route("/keywords/<keyword_name>")
def keywords(keyword_name):
    """Page which give details about a keyword."""
    keyword = Keyword.query.filter_by(name=keyword_name).one_or_none()
    if keyword:
        return render_template('keyword.html', keyword=keyword)
    return redirect_to_user('Keyword {} does not exist.'
                            .format(keyword_name), is_error=True)
