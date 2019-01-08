from flask import Blueprint
from flask import render_template

mod = Blueprint('general', __name__)


@mod.route('/')
def index():
    """Default landing page."""
    return render_template('index.html')


@mod.route("/description")
def ramp():
    """RAMP description request."""
    return render_template('ramp_description.html')
