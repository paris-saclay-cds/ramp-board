import os
# import sys
import codecs
import shutil
import difflib
import logging
import os.path
import time
import tempfile
import datetime
import io
import zipfile
import numpy as np
from flask import request, redirect, url_for, render_template, abort,\
    send_from_directory, session, g, send_file
from flask.ext.login import current_user
from sqlalchemy.orm.exc import NoResultFound
from werkzeug import secure_filename
from wtforms import StringField
from wtforms.widgets import TextArea
from databoard import db
from bokeh.embed import components

import flask
import flask.ext.login as fl
from flask.ext.sqlalchemy import get_debug_queries
from databoard import app, login_manager
import databoard.db_tools as db_tools
import databoard.vizu as vizu
import databoard.config as config
from databoard.model import User, Submission, WorkflowElement,\
    Event, Problem, Keyword, Team, SubmissionFile, UserInteraction,\
    SubmissionSimilarity, EventTeam, DuplicateSubmissionError,\
    TooEarlySubmissionError, MissingExtensionError
from databoard.forms import LoginForm, CodeForm, SubmitForm, ImportForm,\
    UploadForm, UserProfileForm, CreditForm, EmailForm, PasswordForm
from databoard.security import ts


app.secret_key = os.urandom(24)

logger = logging.getLogger('databoard')


@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))


def timestamp_to_time(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp)).strftime(
        '%Y-%m-%d %H:%M:%S')

# TODO: get_auth_token()
# https://flask-login.readthedocs.org/en/latest/#flask.ext.login.LoginManager.user_loader


@app.before_request
def before_request():
    g.user = current_user  # so templates can access user


def check_admin(current_user, event):
    try:
        return (current_user.access_level == 'admin' or
                db_tools.is_admin(event, current_user))
    except AttributeError:
        return False


@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404


@app.route("/login", methods=['GET', 'POST'])
def login():
    db_tools.add_user_interaction(interaction='landing')

    # If there is already a user logged in, don't let another log in
    if current_user.is_authenticated:
        session['logged_in'] = True
        return redirect(url_for('problems'))

    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(name=form.user_name.data).one()
        except NoResultFound:
            flask.flash(u'{} does not exist.'.format(form.user_name.data))
            return flask.redirect(flask.url_for('login'))
        if not db_tools.check_password(
                form.password.data, user.hashed_password):
            flask.flash('Wrong password')
            return flask.redirect(flask.url_for('login'))
        fl.login_user(user, remember=True)  # , remember=form.remember_me.data)
        session['logged_in'] = True
        user.is_authenticated = True
        db.session.commit()
        logger.info(u'{} is logged in'.format(current_user))
        db_tools.add_user_interaction(interaction='login', user=current_user)
        # next = flask.request.args.get('next')
        # next_is_valid should check if the user has valid
        # permission to access the `next` url
        # if not fl.next_is_valid(next):
        #     return flask.abort(400)

        return flask.redirect(flask.url_for('problems'))

    return render_template(
        'login.html',
        form=form,
    )


@app.route("/sign_up", methods=['GET', 'POST'])
def sign_up():
    if current_user.is_authenticated:
        session['logged_in'] = True
        return redirect(url_for('user'))

    form = UserProfileForm()
    if form.validate_on_submit():
        try:
            user = db_tools.create_user(
                name=form.user_name.data,
                password=form.password.data,
                lastname=form.lastname.data,
                firstname=form.firstname.data,
                email=form.email.data,
                linkedin_url=form.linkedin_url.data,
                twitter_url=form.twitter_url.data,
                facebook_url=form.facebook_url.data,
                google_url=form.google_url.data,
                github_url=form.github_url.data,
                website_url=form.website_url.data,
                bio=form.bio.data,
                is_want_news=form.is_want_news.data,
                access_level='asked')
        except Exception as e:
            flask.flash(u'{}'.format(e), category='Sign-up error')
            return redirect(url_for('sign_up'))
        db_tools.send_register_request_mail(user)
        return flask.redirect(flask.url_for('login'))
    return render_template(
        'sign_up.html',
        form=form,
    )


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/description")
def ramp():
    """
    """
    return render_template('ramp_description.html')


@app.route("/user")
def user():
    if current_user.is_authenticated:
        db_tools.add_user_interaction(
            interaction='looking at user', user=current_user)

        events = Event.query.order_by(Event.public_opening_timestamp.desc())
        event_urls_f_names = [
            (event.name,
             event.problem.title + ', ' + event.title,
             db_tools.is_user_signed_up(event.name, current_user.name),
             db_tools.is_user_asked_sign_up(event.name, current_user.name))
            for event in events
            if db_tools.is_public_event(event, current_user)]

    else:
        events = Event.query.filter_by(is_public=True).all()
        event_urls_f_names = [(
            event.name, event.problem.title + ', ' + event.title, False, False)
            for event in events]
    admin = check_admin(current_user, None)
    return render_template('user.html',
                           event_urls_f_names=event_urls_f_names,
                           admin=admin)


@app.route("/data_domains")
def data_domains():
    keywords = Keyword.query.order_by(Keyword.name)
    return render_template('data_domains.html', keywords=keywords)


@app.route("/teaching")
def teaching():
    return render_template('teaching.html')


@app.route("/data_science_themes")
def data_science_themes():
    keywords = Keyword.query.order_by(Keyword.name)
    return render_template('data_science_themes.html', keywords=keywords)


@app.route("/keywords/<keyword_name>")
def keywords(keyword_name):
    keyword = Keyword.query.filter_by(name=keyword_name).one()
    return render_template('keyword.html', keyword=keyword)


@app.route("/problems")
def problems():
    if current_user.is_authenticated:
        db_tools.add_user_interaction(
            interaction='looking at problems', user=current_user)
    else:
        db_tools.add_user_interaction(
            interaction='looking at problems')

    problems = Problem.query.order_by(Problem.id.desc())
    return render_template('problems.html',
                           problems=problems)


@app.route("/problems/<problem_name>")
def problem(problem_name):
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    if problem:
        if current_user.is_authenticated:
            db_tools.add_user_interaction(
                interaction='looking at problem', user=current_user,
                problem=problem)
        else:
            db_tools.add_user_interaction(
                interaction='looking at problem', problem=problem)
        description_f_name = os.path.join(
            config.ramp_kits_path, problem.name, '{}_starting_kit.html'.format(
                problem_name))  
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        return render_template('problem.html',
                               problem=problem,
                               description=description)
    else:
        return _redirect_to_user(u'Problem {} does not exist.'.format(
            problem.name), is_error=True)


@app.route("/event_plots/<event_name>")
def event_plots(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    # if not db_tools.is_public_event(event, current_user):
    #     return _redirect_to_user(u'{}: no event named "{}"'.format(
    #         current_user, event_name))
    if event:
        p = vizu.score_plot(event)
        script, div = components(p)
        return render_template('event_plots.html',
                               script=script,
                               div=div,
                               event=event)
    else:
        return _redirect_to_user(u'Event {} does not exist.'.format(
            event_name), is_error=True)


def _redirect_to_user(message_str, is_error=True, category=None):
    flask.flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return redirect(url_for('problems'))


def _redirect_to_sandbox(event, message_str, is_error=True, category=None):
    flask.flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return flask.redirect(u'/events/{}/sandbox'.format(event.name))


def _redirect_to_credit(submission_hash, message_str, is_error=True,
                        category=None):
    flask.flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return flask.redirect(u'/credit/{}'.format(submission_hash))


@app.route("/events/<event_name>/sign_up")
@fl.login_required
def sign_up_for_event(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            current_user, event_name))
    db_tools.add_user_interaction(
        interaction='signing up at event', user=current_user, event=event)

    db_tools.ask_sign_up_team(event.name, current_user.name)
    if event.is_controled_signup:
        db_tools.send_sign_up_request_mail(event, current_user)
        return _redirect_to_user(
            "Sign-up request is sent to event admins.", is_error=False,
            category='Request sent')
    else:
        db_tools.sign_up_team(event.name, current_user.name)
        return _redirect_to_sandbox(
            event, u'{} is signed up for {}.'.format(current_user, event),
            is_error=False, category='Successful sign-up')


@app.route("/events/<event_name>/sign_up/<user_name>")
@fl.login_required
def approve_sign_up_for_event(event_name, user_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    user = User.query.filter_by(name=user_name).one_or_none()
    if not current_user.access_level == 'admin' or\
            not db_tools.is_admin(event, current_user):
        return _redirect_to_user(u'Sorry {}, you do not have admin rights'.
                                 format(current_user), is_error=True)
    if not event or not user:
        return _redirect_to_user(u'Oups, no event {} or no user {}.'.
                                 format(event_name, user_name), is_error=True)
    db_tools.sign_up_team(event.name, user.name)
    return _redirect_to_user(
        u'{} is signed up for {}.'.format(user, event),
        is_error=False, category='Successful sign-up')


@app.route("/approve_users", methods=['GET', 'POST'])
@fl.login_required
def approve_users():
    if not current_user.access_level == 'admin':
        return _redirect_to_user(u'Sorry {}, you do not have admin rights'.
                                 format(current_user), is_error=True)
    if request.method == 'GET':
        asked_users = User.query.filter_by(access_level='asked')
        asked_sign_up = EventTeam.query.filter_by(approved=False)
        return render_template('approve.html', asked_users=asked_users,
                               asked_sign_up=asked_sign_up, admin=True)
    elif request.method == 'POST':
        users_to_be_approved = request.form.getlist('approve_users')
        event_teams_to_be_approved = request.form.getlist('approve_event_teams')
        message = "Approve users:\n"
        for asked_user in users_to_be_approved:
            db_tools.approve_user(asked_user)
            message += "%s\n" % asked_user
        message += " ** Approved event_team:\n"
        for asked_id in event_teams_to_be_approved:
            asked_event_team = EventTeam.query.get(int(asked_id))
            db_tools.sign_up_team(asked_event_team.event.name,
                                  asked_event_team.team.name)
            message += "%s\n" % asked_event_team
        return _redirect_to_user(message, is_error=False,
                                 category="Approved users")
    # if not user:
    #     return _redirect_to_user(u'Oups, no user {}.'.format(user_name),
    #                              is_error=True)
    # db_tools.approve_user(user.name)
    # return _redirect_to_user(
    #     u'{} is signed up.'.format(user),
    #     is_error=False, category='Successful sign-up')


@app.route("/sign_up/<user_name>")
@fl.login_required
def approve_user(user_name):
    user = User.query.filter_by(name=user_name).one_or_none()
    if not current_user.access_level == 'admin':
        return _redirect_to_user(u'Sorry {}, you do not have admin rights'.
                                 format(current_user), is_error=True)
    if not user:
        return _redirect_to_user(u'Oups, no user {}.'.format(user_name),
                                 is_error=True)
    db_tools.approve_user(user.name)
    return _redirect_to_user(
        u'{} is signed up.'.format(user),
        is_error=False, category='Successful sign-up')


@app.route("/events/<event_name>")
# @fl.login_required
def user_event(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    # if not db_tools.is_public_event(event, current_user):
    #     return _redirect_to_user(u'{}: no event named "{}"'.format(
    #         current_user, event_name))
    if event:
        if current_user.is_authenticated:
            db_tools.add_user_interaction(
                interaction='looking at event', user=current_user, event=event)
        else:
            db_tools.add_user_interaction(
                interaction='looking at event', event=event)
        description_f_name = os.path.join(
            config.ramp_kits_path, event.problem.name,
            '{}_starting_kit.html'.format(event.problem.name))  
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        admin = check_admin(current_user, event)
        if current_user.is_anonymous:
            approved = False
            asked = False
        else:
            approved = db_tools.is_user_signed_up(
                event_name, current_user.name)
            asked = db_tools.is_user_asked_sign_up(
                event.name, current_user.name)
        return render_template('event.html',
                               description=description,
                               event=event,
                               admin=admin,
                               approved=approved,
                               asked=asked)
    else:
        return _redirect_to_user(u'Event {} does not exist.'.format(
            event_name), is_error=True)


@app.route("/events/<event_name>/starting_kit")
@fl.login_required
def download_starting_kit(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    starting_kit_path = os.path.abspath(os.path.join(
        config.root_path, config.problems_d_name, event.problem.name))
    f_name = u'{}.zip'.format(config.sandbox_d_name)
    print(starting_kit_path)
    return send_from_directory(
        starting_kit_path, f_name, as_attachment=True,
        attachment_filename=u'{}_{}'.format(event_name, f_name),
        mimetype='application/octet-stream')


@app.route("/problems/<problem_name>/starting_kit")
def download_problem_starting_kit(problem_name):
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    starting_kit_path = os.path.abspath(os.path.join(
        config.root_path, config.problems_d_name, problem.name))
    f_name = u'{}.zip'.format(config.sandbox_d_name)
    print(starting_kit_path)
    return send_from_directory(
        starting_kit_path, f_name, as_attachment=True,
        attachment_filename=u'{}_{}'.format(problem_name, f_name),
        mimetype='application/octet-stream')


@app.route("/events/<event_name>/my_submissions")
@fl.login_required
def my_submissions(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            current_user, event_name))
    db_tools.add_user_interaction(
        interaction='looking at my_submissions',
        user=current_user, event=event)
    # Doesn't work if team mergers are allowed
    team = Team.query.filter_by(name=current_user.name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()

    leaderboard_html = event_team.leaderboard_html
    failed_leaderboard_html = event_team.failed_leaderboard_html
    new_leaderboard_html = event_team.new_leaderboard_html
    admin = check_admin(current_user, event)
    return render_template('leaderboard.html',
                           leaderboard_title='Trained submissions',
                           leaderboard=leaderboard_html,
                           failed_leaderboard=failed_leaderboard_html,
                           new_leaderboard=new_leaderboard_html,
                           event=event,
                           admin=admin)


@app.route("/events/<event_name>/leaderboard")
@fl.login_required
def leaderboard(event_name):
    # start = time.time()

    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            current_user, event_name))
    db_tools.add_user_interaction(
        interaction='looking at leaderboard',
        user=current_user, event=event)

    # logger.info(u'leaderboard user_interaction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))
    # start = time.time()

    if db_tools.is_open_leaderboard(event, current_user):
        leaderboard_html = event.public_leaderboard_html_with_links
    else:
        leaderboard_html = event.public_leaderboard_html_no_links

    # logger.info(u'leaderboard db access takes {}ms'.format(
    #     int(1000 * (time.time() - start))))
    # start = time.time()

    leaderboard_kwargs = dict(
        leaderboard=leaderboard_html,
        leaderboard_title='Leaderboard',
        event=event
    )

    if current_user.access_level == 'admin' or\
            db_tools.is_admin(event, current_user):
        failed_leaderboard_html = event.failed_leaderboard_html
        new_leaderboard_html = event.new_leaderboard_html
        template = render_template(
            'leaderboard.html',
            failed_leaderboard=failed_leaderboard_html,
            new_leaderboard=new_leaderboard_html,
            admin=True,
            **leaderboard_kwargs)
    else:
        template = render_template(
            'leaderboard.html',
            **leaderboard_kwargs)

    # logger.info(u'leaderboard rendering takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    return template

@app.route("/<submission_hash>/<f_name>", methods=['GET', 'POST'])
@fl.login_required
def view_model(submission_hash, f_name):
    """Rendering submission codes using templates/submission.html.

    The code of
    f_name is displayed in the left panel, the list of submissions files
    is in the right panel. Clicking on a file will show that file (using
    the same template). Clicking on the name on the top will download the file
    itself (managed in the template). Clicking on "Archive" will zip all
    the submission files and download them (managed here).


    Parameters
    ----------
    event_name : string
        The team.name of the submission.
    submission_hash : string
        The hash_ of the submission.
    f_name : string
        The name of the submission file

    Returns
    -------
     : html string
        The rendered submission.html page.
    """
    submission = Submission.query.filter_by(
        hash_=submission_hash).one_or_none()
    if submission is None:
        error_str = u'Missing submission {}: {}/{}'.format(
            current_user, submission_hash, f_name)
        return _redirect_to_user(error_str)
    event = submission.event_team.event
    if not db_tools.is_open_code(event, current_user, submission):
        error_str = u'{} has no permission to look at {}/{}/{}\n'.format(
            current_user, event, submission, f_name)
        error_str += u'The code links will open at (UTC) {}'.format(
            db_tools.date_time_format(event.public_opening_timestamp))
        return _redirect_to_user(error_str)
    team = submission.event_team.team
    workflow_element_name = f_name.split('.')[0]
    workflow_element = WorkflowElement.query.filter_by(
        name=workflow_element_name, workflow=event.workflow).one_or_none()
    if workflow_element is None:
        error_str = u'{} is not a valid workflow element by {} in {}/{}/{}/{}'.\
            format(workflow_element_name, current_user, event, team,
                   submission, f_name)
        return _redirect_to_user(error_str)
    submission_file = SubmissionFile.query.filter_by(
        submission=submission, workflow_element=workflow_element).one_or_none()
    if submission_file is None:
        error_str = u'No submission file by {} in {}/{}/{}/{}'.format(
            current_user, event, team, submission, f_name)
        return _redirect_to_user(error_str)

    # superfluous, perhaps when we'll have different extensions?
    f_name = submission_file.f_name

    submission_abspath = os.path.abspath(submission.path)
    if not os.path.exists(submission_abspath):
        error_str = u'{} does not exist by {} in {}/{}/{}/{}'.format(
            submission_abspath, current_user, event, team, submission, f_name)
        return _redirect_to_user(error_str)

    db_tools.add_user_interaction(
        interaction='looking at submission', user=current_user, event=event,
        submission=submission, submission_file=submission_file)

    logger.info(u'{} is looking at {}/{}/{}/{}'.format(
        current_user, event, team, submission, f_name))

    # Downloading file if it is not editable (e.g., external_data.csv)
    if not workflow_element.is_editable:
        # archive_filename = f_name  + '.zip'
        # with changedir(submission_abspath):
        #    with ZipFile(archive_filename, 'w') as archive:
        #        archive.write(f_name)
        db_tools.add_user_interaction(
            interaction='download', user=current_user, event=event,
            submission=submission, submission_file=submission_file)

        return send_from_directory(
            submission_abspath, f_name, as_attachment=True,
            attachment_filename=u'{}_{}'.format(submission.hash_[:6], f_name),
            mimetype='application/octet-stream')

    # Importing selected files into sandbox
    choices = [(f, f) for f in submission.f_names]
    import_form = ImportForm()
    import_form.selected_f_names.choices = choices
    if import_form.validate_on_submit():
        sandbox_submission = db_tools.get_sandbox(event, current_user)
        for f_name in import_form.selected_f_names.data:
            logger.info(u'{} is importing {}/{}/{}/{}'.format(
                current_user, event, team, submission, f_name))

            # TODO: deal with different extensions of the same file
            src = os.path.join(submission.path, f_name)
            dst = os.path.join(sandbox_submission.path, f_name)
            shutil.copy2(src, dst)  # copying also metadata
            logger.info(u'Copying {} to {}'.format(src, dst))

            workflow_element = WorkflowElement.query.filter_by(
                name=f_name.split('.')[0], workflow=event.workflow).one()
            submission_file = SubmissionFile.query.filter_by(
                submission=submission,
                workflow_element=workflow_element).one()
            db_tools.add_user_interaction(
                interaction='copy', user=current_user, event=event,
                submission=submission, submission_file=submission_file)

        return flask.redirect(u'/events/{}/sandbox'.format(event.name))

    with open(os.path.join(submission.path, f_name)) as f:
        code = f.read()
    admin = check_admin(current_user, event)
    return render_template(
        'submission.html',
        event=event,
        code=code,
        submission=submission,
        f_name=f_name,
        import_form=import_form,
        admin=admin)


@app.route("/download/<submission_hash>")
@fl.login_required
def download_submission(submission_hash):
    submission = Submission.query.filter_by(hash_=submission_hash).one_or_none()
    if submission is None:
        error_str = u'Missing submission: {}'.format(submission_hash)
        return _redirect_to_user(error_str)
    event = submission.event_team.event
    if not db_tools.is_open_code(event, current_user, submission):
        error_str = u'{} has no right to look at {}/{}'.format(
            current_user, event, submission)
        return _redirect_to_user(error_str)
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        files = submission.files
        for ff in files:
            data = zipfile.ZipInfo(ff.f_name)
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, ff.get_code())
    memory_file.seek(0)
    return send_file(memory_file,
                     attachment_filename='submission_%s.zip' % submission.id,
                     as_attachment=True)


@app.route("/events/<event_name>/sandbox", methods=['GET', 'POST'])
@fl.login_required
def sandbox(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, current_user):
        return _redirect_to_user(u'{}: no access or no event named "{}"'.
                                 format(current_user, event_name))
    if not db_tools.is_open_code(event, current_user):
        error_str = u'No access to sandbox for event {}. '\
            u'If you have already signed up, please wait for approval'.\
            format(event)
        return _redirect_to_user(error_str)

    sandbox_submission = db_tools.get_sandbox(event, current_user)
    event_team = db_tools.get_active_user_event_team(event, current_user)

    # The amount of python magic we have to do for rendering a variable number
    # of textareas, named and populated at run time, is mind boggling.

    # First we need to make sure CodeForm is empty
    # for name_code in CodeForm.names_codes:
    #     name, _ = name_code
    #     delattr(CodeForm, name)
    CodeForm.names_codes = []

    # Then we create named
    # fields in the CodeForm class for each editable submission file. They have
    # to be populated when the code_form object is created, so we also
    # create a code_form_kwargs dictionary and populate it with the codes.
    code_form_kwargs = {}
    for submission_file in sandbox_submission.files:
        if submission_file.is_editable:
            f_field = submission_file.name
            setattr(CodeForm, f_field, StringField(u'Text', widget=TextArea()))
            code_form_kwargs[f_field] = submission_file.get_code()
    code_form = CodeForm(**code_form_kwargs)
    # Then, to be able to iterate over the files in the sandbox.html template,
    # we also fill a separate table of pairs (file name, code). The text areas
    # in the template will then have to be created manually.
    for submission_file in sandbox_submission.files:
        if submission_file.is_editable:
            code_form.names_codes.append(
                (submission_file.name, submission_file.get_code()))

    submit_form = SubmitForm(submission_name=event_team.last_submission_name)
    upload_form = UploadForm()

    if len(code_form.names_codes) > 0 and code_form.validate_on_submit()\
            and getattr(code_form, code_form.names_codes[0][0]).data:
        try:
            for submission_file in sandbox_submission.files:
                if submission_file.is_editable:
                    old_code = submission_file.get_code()
                    submission_file.set_code(
                        getattr(code_form, submission_file.name).data)
                    new_code = submission_file.get_code()
                    diff = '\n'.join(difflib.unified_diff(
                        old_code.splitlines(), new_code.splitlines()))
                    similarity = difflib.SequenceMatcher(
                        a=old_code, b=new_code).ratio()
                    db_tools.add_user_interaction(
                        interaction='save', user=current_user, event=event,
                        submission_file=submission_file,
                        diff=diff, similarity=similarity)
        except Exception as e:
            return _redirect_to_sandbox(event, u'Error: {}'.format(e))
        return _redirect_to_sandbox(
            event, u'{} saved submission files for {}.'.format(
                current_user, event_team, event),
            is_error=False, category='File saved')

    if upload_form.validate_on_submit() and upload_form.file.data:
        upload_f_name = secure_filename(upload_form.file.data.filename)
        upload_name = upload_f_name.split('.')[0]
        upload_workflow_element = WorkflowElement.query.filter_by(
            name=upload_name, workflow=event.workflow).one_or_none()
        if upload_workflow_element is None:
            return _redirect_to_sandbox(
                event, u'{} is not in the file list.'.format(upload_f_name))

        submission_file = SubmissionFile.query.filter_by(
            submission=sandbox_submission,
            workflow_element=upload_workflow_element).one()
        if submission_file.is_editable:
            old_code = submission_file.get_code()

        tmp_f_name = os.path.join(tempfile.gettempdir(), upload_f_name)
        upload_form.file.data.save(tmp_f_name)
        file_length = os.stat(tmp_f_name).st_size
        if upload_workflow_element.max_size is not None and\
                file_length > upload_workflow_element.max_size:
            return _redirect_to_sandbox(
                event, u'File is too big: {} exceeds max size {}'.format(
                    file_length, upload_workflow_element.max_size))
        if submission_file.is_editable:
            try:
                with open(tmp_f_name) as f:
                    code = f.read()
                    submission_file.set_code(code)  # to verify eg asciiness
            except Exception as e:
                return _redirect_to_sandbox(event, u'Error: {}'.format(e))
        else:
            # non-editable files are not verified for now
            dst = os.path.join(sandbox_submission.path, upload_f_name)
            shutil.copy2(tmp_f_name, dst)
        logger.info(u'{} uploaded {} in {}'.format(
            current_user, upload_f_name, event))

        if submission_file.is_editable:
            new_code = submission_file.get_code()
            diff = '\n'.join(difflib.unified_diff(
                old_code.splitlines(), new_code.splitlines()))
            similarity = difflib.SequenceMatcher(
                a=old_code, b=new_code).ratio()
            db_tools.add_user_interaction(
                interaction='upload', user=current_user, event=event,
                submission_file=submission_file,
                diff=diff, similarity=similarity)
        else:
            db_tools.add_user_interaction(
                interaction='upload', user=current_user, event=event,
                submission_file=submission_file)

        return flask.redirect(request.referrer)
        # TODO: handle different extensions for the same workflow element
        # ie: now we let upload eg external_data.bla, and only fail at
        # submission, without giving a message

    if submit_form.validate_on_submit() and submit_form.submission_name.data:
        new_submission_name = submit_form.submission_name.data
        if len(new_submission_name) < 4 or len(new_submission_name) > 20:
            return _redirect_to_sandbox(
                event, 'Submission name should have length between 4 and 20 ' +
                'characters.')
        try:
            new_submission_name.encode('ascii')
        except Exception as e:
            return _redirect_to_sandbox(event, u'Error: {}'.format(e))

        try:
            new_submission = db_tools.make_submission_and_copy_files(
                event.name, event_team.team.name, new_submission_name,
                sandbox_submission.path)
        except DuplicateSubmissionError:
            return _redirect_to_sandbox(
                event, u'Submission {} already exists. Please change the name.'
                .format(new_submission_name))
        except MissingExtensionError as e:
            return _redirect_to_sandbox(
                event, u'Missing extension, {}'.format(e.value))
        except TooEarlySubmissionError as e:
            return _redirect_to_sandbox(event, e.value)

        logger.info(u'{} submitted {} for {}.'.format(
            current_user, new_submission.name, event_team))
        if event.is_send_submitted_mails:
            try:
                db_tools.send_submission_mails(
                    current_user, new_submission, event_team)
            except Exception as e:
                logger.error(u'mail was not sent {} submitted {} for {}\n{}.'.format(
                    current_user, new_submission.name, event_team, e))
        flask.flash(u'{} submitted {} for {}.'.format(
            current_user, new_submission.name, event_team),
            category='Submission')

        db_tools.add_user_interaction(
            interaction='submit', user=current_user, event=event,
            submission=new_submission)

        return flask.redirect(u'/credit/{}'.format(new_submission.hash_))
    admin = check_admin(current_user, event)
    return render_template('sandbox.html',
                           submission_names=sandbox_submission.f_names,
                           code_form=code_form,
                           submit_form=submit_form,
                           upload_form=upload_form,
                           event=event,
                           admin=admin)


@app.route("/credit/<submission_hash>", methods=['GET', 'POST'])
@fl.login_required
def credit(submission_hash):
    submission = Submission.query.filter_by(
        hash_=submission_hash).one_or_none()
    if submission is None:
        error_str = u'Missing submission {}: {}'.format(
            current_user, submission_hash)
        return _redirect_to_user(error_str)
    event_team = submission.event_team
    event = event_team.event
    if not db_tools.is_open_code(event, current_user, submission):
        error_str = u'{} has no right to look at {}/{}'.format(
            current_user, event, submission)
        return _redirect_to_user(error_str)
    source_submissions = db_tools.get_source_submissions(submission)

    def get_s_field(source_submission):
        return u'{}/{}/{}'.format(
            source_submission.event_team.event.name,
            source_submission.event_team.team.name,
            source_submission.name)

    # Make sure that CreditForm is empty
    CreditForm.name_credits = []
    credit_form_kwargs = {}
    for source_submission in source_submissions:
        s_field = get_s_field(source_submission)
        setattr(CreditForm, s_field, StringField(u'Text'))
    credit_form = CreditForm(**credit_form_kwargs)
    sum_credit = 0
    new = True
    for source_submission in source_submissions:
        s_field = get_s_field(source_submission)
        submission_similaritys = SubmissionSimilarity.query.filter_by(
            type='target_credit', user=current_user,
            source_submission=source_submission,
            target_submission=submission).all()
        if not submission_similaritys:
            credit = 0
        else:
            new = False
            # find the last credit (in case crediter changes her mind)
            submission_similaritys.sort(
                key=lambda x: x.timestamp, reverse=True)
            credit = int(round(100 * submission_similaritys[0].similarity))
            sum_credit += credit
        credit_form.name_credits.append(
            (s_field, str(credit), source_submission.link))
    # This doesnt work, not sure why
    # if not new:
    #    credit_form.self_credit.data = str(100 - sum_credit)
    if credit_form.validate_on_submit():
        try:
            sum_credit = int(credit_form.self_credit.data)
            for source_submission in source_submissions:
                s_field = get_s_field(source_submission)
                sum_credit += int(getattr(credit_form, s_field).data)
            if sum_credit != 100:
                return _redirect_to_credit(
                    submission_hash,
                    'Error: The total credit should add up to 100')
        except Exception as e:
            return _redirect_to_credit(submission_hash, u'Error: {}'.format(e))
        for source_submission in source_submissions:
            s_field = get_s_field(source_submission)
            similarity = int(getattr(credit_form, s_field).data) / 100.
            submission_similarity = SubmissionSimilarity.query.filter_by(
                type='target_credit', user=current_user,
                source_submission=source_submission,
                target_submission=submission).all()
            # if submission_similarity is not empty, we need to 
            # add zero to cancel previous credits explicitly
            if similarity > 0 or submission_similarity:
                submission_similarity = SubmissionSimilarity(
                    type='target_credit', user=current_user,
                    source_submission=source_submission,
                    target_submission=submission,
                    similarity=similarity,
                    timestamp=datetime.datetime.utcnow())
                db.session.add(submission_similarity)
        db.session.commit()

        db_tools.add_user_interaction(
            interaction='giving credit', user=current_user, event=event,
            submission=submission)

        return flask.redirect(u'/events/{}/sandbox'.format(event.name))

    admin = check_admin(current_user, event)
    return render_template('credit.html',
                           submission=submission,
                           source_submissions=source_submissions,
                           credit_form=credit_form,
                           event=event,
                           admin=admin)


@app.route("/logout")
@fl.login_required
def logout():
    user = current_user
    db_tools.add_user_interaction(interaction='logout', user=user)
    session['logged_in'] = False
    user.is_authenticated = False
    db.session.commit()
    logger.info(u'{} is logged out'.format(user))
    fl.logout_user()

    return redirect(flask.url_for('login'))


# @app.route("/teams/<team_name>")
# @fl.login_required
# def team(team_name):
#     return render_template('team.html',
#                            ramp_title=config.config_object.specific.ramp_title)


@app.route("/events/<event_name>/private_leaderboard")
@fl.login_required
def private_leaderboard(event_name):
    # start = time.time()

    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            current_user, event_name))
    if (not db_tools.is_admin(event, current_user) and
        (event.closing_timestamp is None or
            event.closing_timestamp > datetime.datetime.utcnow())):
        return redirect(url_for('user'))

    db_tools.add_user_interaction(
        interaction='looking at private leaderboard',
        user=current_user, event=event)
    leaderboard_html = event.private_leaderboard_html
    admin = check_admin(current_user, event)
    template = render_template(
        'leaderboard.html',
        leaderboard_title='Leaderboard',
        leaderboard=leaderboard_html,
        event=event,
        private=True,
        admin=admin
    )

    # logger.info(u'private leaderboard takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    return template


@app.route("/<submission_hash>/error.txt")
@fl.login_required
def view_submission_error(submission_hash):
    """Rendering submission codes using templates/submission.html.

    The code of
    f_name is displayed in the left panel, the list of submissions files
    is in the right panel. Clicking on a file will show that file (using
    the same template). Clicking on the name on the top will download the file
    itself (managed in the template). Clicking on "Archive" will zip all
    the submission files and download them (managed here).


    Parameters
    ----------
    team_name : string
        The team.name of the submission.
    submission_hash : string
        The hash_ of the submission.

    Returns
    -------
     : html string
        The rendered submission_error.html page.
    """
    submission = Submission.query.filter_by(hash_=submission_hash).one()
    if submission is None:
        error_str = u'Missing submission {}: {}'.format(
            current_user, submission_hash)
        return _redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    # TODO: check if event == submission.event_team.event

    db_tools.add_user_interaction(
        interaction='looking at error', user=current_user, event=event,
        submission=submission)

    return render_template(
        'submission_error.html',
        submission=submission,
        team=team,
        event=event
    )


@app.route("/user_interactions")
@fl.login_required
def user_interactions():
    if not current_user.is_authenticated\
            or current_user.access_level != 'admin':
        return redirect(url_for('login'))
    user_interactions_html = db_tools.get_user_interactions()
    return render_template(
        'user_interactions.html',
        user_interactions_title='User interactions',
        user_interactions=user_interactions_html
    )


@app.route("/submissions/diff_bef24208a45043059/<id>")
@fl.login_required
def submission_file_diff(id):
    if not current_user.is_authenticated\
            or current_user.access_level != 'admin':
        return redirect(url_for('login'))
    user_interaction = UserInteraction.query.filter_by(id=id).one()
    return render_template(
        'submission_file_diff.html',
        diff=user_interaction.submission_file_diff
    )


@app.after_request
def after_request(response):
    for query in get_debug_queries():
        if query.duration >= config.DATABASE_QUERY_TIMEOUT:
            app.logger.warning("SLOW QUERY: %s\nParameters: %s\n"
                               "Duration: %fs\nContext: %s\n"
                               % (query.statement, query.parameters,
                                  query.duration, query.context))
    return response


@app.route("/events/<event_name>/dashboard_submissions")
@fl.login_required
def dashboard_submissions(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()

    if current_user.access_level == 'admin' or\
            db_tools.is_admin(event, current_user):
        # Get dates and number of submissions
        submissions_ = db.session.query(Submission, Event, EventTeam).\
            filter(Event.name == event.name).\
            filter(Event.id == EventTeam.event_id).\
            filter(EventTeam.id == Submission.event_team_id).\
            order_by(Submission.submission_timestamp).all()
        if submissions_:
            submissions = list(zip(*submissions_)[0])
        else:
            submissions = []
        submissions = [submission for submission in submissions
                       if submission.name != config.sandbox_d_name]
        timestamp_submissions = [submission.submission_timestamp.
                                 strftime('%Y-%m-%d %H:%M:%S')
                                 for submission in submissions]
        name_submissions = [submission.name for submission in submissions]
        cumulated_submissions = range(1, 1 + len(submissions))
        training_sec = [(submission.training_timestamp -
                         submission.submission_timestamp).seconds / 60.
                        if submission.training_timestamp is not None
                        else 0
                        for submission in submissions]
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
    else:
        return _redirect_to_user(
            u'Sorry {}, you do not have admin access for {}"'.
            format(current_user, event_name))


@app.route('/reset_password', methods=["GET", "POST"])
def reset_password():
    form = EmailForm()
    error = ''
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.access_level != 'asked':
            subject = "Password reset requested - RAMP website"

            # Here we use the URLSafeTimedSerializer we created in security
            token = ts.dumps(user.email, salt='recover-key')

            recover_url = url_for(
                'reset_with_token',
                token=token,
                _external=True)

            header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
                user.email, config.MAIL_USERNAME, subject)
            body = ('Hi %s, \n\nclick on the link to reset your password:\n' %
                    user.firstname.encode('utf-8'))
            body += recover_url
            body += '\n\nSee you on the RAMP website!'
            db_tools.send_mail(user.email, subject, header + body)
            logger.info(
                'Password reset requested for user {}'.format(user.name))
            logger.info(recover_url)
            return redirect(url_for('login'))
        else:
            error = ('Sorry, but this user was not approved or the email was '
                     'wrong. If you need some help, send an email to %s' %
                     config.MAIL_USERNAME)
    return render_template('reset_password.html', form=form, error=error)


@app.route('/reset/<token>', methods=["GET", "POST"])
def reset_with_token(token):
    try:
        email = ts.loads(token, salt="recover-key", max_age=86400)
    except:
        abort(404)

    form = PasswordForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=email).first_or_404()

        user.hashed_password = db_tools.get_hashed_password(form.password.data)

        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('reset_with_token.html', form=form, token=token)
