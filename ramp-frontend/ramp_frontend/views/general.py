from flask import Blueprint
from flask import render_template
import os as os

from ramp_database.model import Keyword

from .redirect import redirect_to_user

mod = Blueprint('general', __name__)


@mod.route('/')
def index():
    """Default landing page."""
    print('HERE I AM:')
    print(os.getcwd())
    images = os.listdir('ramp-frontend/ramp_frontend/static/img/powered_by/')
    return render_template('index.html', images=images)


@mod.route("/description")
def ramp():
    """RAMP description request."""
    return render_template('ramp_description.html')


@mod.route("/data_domains")
def data_domains():
    """Review of all possible keyword attached to the different RAMP
    problems."""
    current_keywords = Keyword.query.order_by(Keyword.name)
    return render_template('data_domains.html',
                           keywords=current_keywords)


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


@mod.route("/")
def get_logos():
    images = os.listdir('../static/img/powered_by/')
    print('images:')
    print(images)
    return render_template('index.html', images=images)
