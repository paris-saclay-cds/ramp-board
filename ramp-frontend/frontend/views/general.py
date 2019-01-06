from flask import  Blueprint
from flask import render_template

mod = Blueprint('general', __name__)


@mod.route('/')
def index():
    return render_template('index.html')
