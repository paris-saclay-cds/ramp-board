import os
# import sys
# import shutil
import logging
import os.path
import datetime

from git import Repo
from zipfile import ZipFile
from flask import request, redirect, url_for, render_template,\
    send_from_directory, flash, session, g
from flask.ext.login import current_user, login_required
from sqlalchemy.orm.exc import NoResultFound

import flask
import flask.ext.login as fl
from databoard import app, db, login_manager
from databoard.generic import changedir
from databoard.config import repos_path
import databoard.config as config
import databoard.db_tools as db_tools
from databoard.model import User, Team, Submission
from databoard.forms import LoginForm
#from app import app, db, lm, oid

app.secret_key = os.urandom(24)

logger = logging.getLogger('databoard')


@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


def model_with_link(path_model):
    print path_model
    path, model, listing = path_model.split('+++++!*****')

    filename = listing.split('|')[0]
    filename_path = '/models/{}/{}'.format(path, filename)

    # if the tag name is too long, shrink it.
    if len(model) > tag_len_limit:
        model_trucated = model[:tag_len_limit] + '[...]'
        link = '<a href="{0}" class="popup" data-content="{1}">{2}</a>'.format(
            filename_path, model, model_trucated)
    else:
        link = '<a href="{0}">{1}</a>'.format(filename_path, model)
    return link


def error_local_to_url(path):
    filename = '/models/%s/error' % (path)
    link = '<a href="{0}">Error</a>'.format(filename)
    return link


def timestamp_to_time(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp)).strftime(
        '%Y-%m-%d %H:%M:%S')

# TODO: get_auth_token() 
# https://flask-login.readthedocs.org/en/latest/#flask.ext.login.LoginManager.user_loader


@app.before_request
def before_request():
    g.user = current_user  # so templates can access user


@app.route("/", methods=['GET', 'POST'])
@app.route("/login", methods=['GET', 'POST'])
def login():
    # If there is already a user logged in, don't let another log in
    if current_user.is_authenticated:
        session['logged_in'] = True
        return redirect(url_for('user'))

    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(name=form.user_name.data).one()
        except NoResultFound:
            flask.flash('{} does not exist.'.format(form.user_name.data))
            return flask.redirect(flask.url_for('login'))
        if not db_tools.check_password(
                form.password.data, user.hashed_password):
            flask.flash('Wrong password')
            return flask.redirect(flask.url_for('login'))
        fl.login_user(user)  # , remember=form.remember_me.data)
        session['logged_in'] = True
        logger.info('{} is logged in'.format(current_user))
        # next = flask.request.args.get('next')
        # next_is_valid should check if the user has valid
        # permission to access the `next` url
        #if not fl.next_is_valid(next):
        #    return flask.abort(400)

        return flask.redirect(flask.url_for('user'))
    return render_template('login.html',
                           ramp_title=config.config_object.specific.ramp_title,
                           form=form)


@app.route("/user", methods=['GET', 'POST'])
@fl.login_required
def user():
    leaderbord_html = db_tools.get_public_leaderboard(user=current_user)
    failed_submissions_html = db_tools.get_failed_submissions(
        user=current_user)
    return render_template('leaderboard.html',
                           leaderboard=leaderbord_html,
                           failed_submissions=failed_submissions_html,
                           ramp_title=config.config_object.specific.ramp_title)


@app.route("/logout")
@fl.login_required
def logout():
    session['logged_in'] = False
    logger.info('{} is logged out'.format(current_user))
    fl.logout_user()
    return redirect(flask.url_for('login'))


# @app.route("/teams/<team_name>")
# @fl.login_required
# def team(team_name):
#     return render_template('team.html',
#                            ramp_title=config.config_object.specific.ramp_title)


@app.route("/leaderboard")
def show_leaderboard():
    leaderbord_html = db_tools.get_public_leaderboard()
    return render_template(
        'leaderboard.html', leaderboard=leaderbord_html,
        ramp_title=config.config_object.specific.ramp_title)


# TODO: private leaderboard


@app.route("/submissions/<team_name>/<summission_hash>/<f_name>")
@app.route("/submissions/<team_name>/<summission_hash>/<f_name>/raw")
def view_model(team_name, summission_hash, f_name):
    """Rendering submission codes using templates/submission.html. The code of
    f_name is displayed in the left panel, the list of submissions files
    is in the right panel. Clicking on a file will show that file (using
    the same template). Clicking on the name on the top will download the file
    itself (managed in the template). Clicking on "Archive" will zip all
    the submission files and download them (managed here).


    Parameters
    ----------
    team_name : string
        The team.name of the submission.
    summission_hash : string
        The hash_ of the submission.
    f_name : string
        The name of the submission file

    Returns
    -------
     : html string
        The rendered submission.html page.
    """
    specific = config.config_object.specific

    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        team=team, hash_=summission_hash).one()
    submission_abspath = os.path.abspath(submission.path)
    archive_filename = 'archive.zip'

    if request.path.split('/')[-1] == 'raw':
        with changedir(submission_abspath):
            with ZipFile(archive_filename, 'w') as archive:
                for submission_file in submission.submission_files:
                    archive.write(submission_file.name)

        return send_from_directory(
            submission_abspath, f_name, as_attachment=True,
            attachment_filename='{}_{}_{}'.format(
                team_name, summission_hash[:6], f_name),
            mimetype='application/octet-stream')

    archive_url = '/submissions/{}/{}/{}/raw'.format(
        team_name, summission_hash, os.path.basename(archive_filename))

    submission_url = request.path.rstrip('/') + '/raw'
    submission_f_name = os.path.join(submission_abspath, f_name)
    if not os.path.exists(submission_f_name):
        return redirect(url_for('show_leaderboard'))

    with open(submission_f_name) as f:
        code = f.read()

    return render_template(
        'submission.html',
        code=code,
        submission_url=submission_url,
        submission_f_names=submission.f_names,
        archive_url=archive_url,
        f_name=f_name,
        submission_name=submission.name,
        team_name=team.name,
        ramp_title=specific.ramp_title)


@app.route("/submissions/<team_name>/<summission_hash>/error.txt")
def view_submission_error(team_name, summission_hash):
    """Rendering submission codes using templates/submission.html. The code of
    f_name is displayed in the left panel, the list of submissions files
    is in the right panel. Clicking on a file will show that file (using
    the same template). Clicking on the name on the top will download the file
    itself (managed in the template). Clicking on "Archive" will zip all
    the submission files and download them (managed here).


    Parameters
    ----------
    team_name : string
        The team.name of the submission.
    summission_hash : string
        The hash_ of the submission.

    Returns
    -------
     : html string
        The rendered submission_error.html page.
    """
    specific = config.config_object.specific

    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        team=team, hash_=summission_hash).one()
    submission_abspath = os.path.abspath(submission.path)

    submission_url = request.path.rstrip('/') + '/raw'

    return render_template(
        'submission_error.html',
        error_msg=submission.error_msg,
        submission_state=submission.state,
        submission_url=submission_url,
        submission_f_names=submission.f_names,
        submission_name=submission.name,
        team_name=team.name,
        ramp_title=specific.ramp_title)


@app.route("/add/", methods=("POST",))
def add_team_repo():
    if request.method == "POST":
        repo_name = request.form["name"].strip()
        repo_path = request.form["url"].strip()
        message = ''

        correct_name = True
        for name_part in repo_name.split('_'):
            if not name_part.isalnum():
                correct_name = False
                message = 'Incorrect team name. Please only use letters, digits and underscores.'
                break

        if correct_name:
            try:
                Repo.clone_from(
                    repo_path, os.path.join(repos_path, repo_name))
            except Exception as e:
                logger.error('Unable to add a repository: \n{}'.format(e))
                message = str(e)

        if message:
            flash(message)

        return redirect(url_for('list_teams_repos'))
