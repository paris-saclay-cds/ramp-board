from __future__ import print_function, division, absolute_import

import os
import time
import codecs
import shutil
import difflib
import logging
import tempfile
import datetime
import io
import zipfile
from flask import (
    request, redirect, url_for, render_template, abort, send_from_directory,
    session, g, send_file, flash)
from sqlalchemy.orm.exc import NoResultFound
from werkzeug import secure_filename
from wtforms import StringField
from wtforms.widgets import TextArea

import flask_login as fl
import flask_sqlalchemy as fs
from sqlalchemy.exc import IntegrityError

from databoard import db
from databoard import app, login_manager

from . import db_tools
from . import vizu
from . import config
from .model import (
    User, Submission, WorkflowElement, Event, Problem, Keyword, Team,
    SubmissionFile, UserInteraction, SubmissionSimilarity, EventTeam,
    DuplicateSubmissionError, TooEarlySubmissionError, MissingExtensionError,
    NameClashError)
from .forms import (
    LoginForm, CodeForm, SubmitForm, ImportForm, UploadForm,
    UserCreateProfileForm, UserUpdateProfileForm,
    CreditForm, EmailForm, PasswordForm, EventUpdateProfileForm,
    AskForEventForm)
from .security import ts


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
    g.user = fl.current_user  # so templates can access user


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
    if fl.current_user.is_authenticated:
        session['logged_in'] = True
        return redirect(url_for('problems'))

    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(name=form.user_name.data).one()
        except NoResultFound:
            flash(u'{} does not exist.'.format(form.user_name.data))
            return redirect(url_for('login'))
        if not db_tools.check_password(
                form.password.data, user.hashed_password):
            flash('Wrong password')
            return redirect(url_for('login'))
        fl.login_user(user, remember=True)  # , remember=form.remember_me.data)
        session['logged_in'] = True
        user.is_authenticated = True
        db.session.commit()
        logger.info(u'{} is logged in'.format(fl.current_user.name))
        db_tools.add_user_interaction(
            interaction='login', user=fl.current_user)
        next = request.args.get('next')
        if next is None:
            next = url_for('problems')
        return redirect(next)

    return render_template(
        'login.html',
        form=form,
    )


@app.route("/sign_up", methods=['GET', 'POST'])
def sign_up():
    if fl.current_user.is_authenticated:
        session['logged_in'] = True
        return redirect(url_for('problems'))

    form = UserCreateProfileForm()
    if form.validate_on_submit():
        if form.linkedin_url.data != 'http://doxycycline-cheapbuy.site/':
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
                flash(u'{}'.format(e), category='Sign-up error')
                return redirect(url_for('sign_up'))
            db_tools.send_register_request_mail(user)
        return redirect(url_for('login'))
    return render_template(
        'sign_up.html',
        form=form,
    )


@app.route("/update_profile", methods=['GET', 'POST'])
@fl.login_required
def update_profile():
    form = UserUpdateProfileForm()
    form.user_name.data = fl.current_user.name
    if form.validate_on_submit():
        try:
            db_tools.update_user(fl.current_user, form)
        except Exception as e:
            flash(u'{}'.format(e), category='Update profile error')
            return redirect(url_for('update_profile'))
        # db_tools.send_register_request_mail(user)
        return redirect(url_for('problems'))
    form.lastname.data = fl.current_user.lastname
    form.firstname.data = fl.current_user.firstname
    form.email.data = fl.current_user.email
    form.linkedin_url.data = fl.current_user.linkedin_url
    form.twitter_url.data = fl.current_user.twitter_url
    form.facebook_url.data = fl.current_user.facebook_url
    form.google_url.data = fl.current_user.google_url
    form.github_url.data = fl.current_user.github_url
    form.website_url.data = fl.current_user.website_url
    form.bio.data = fl.current_user.bio
    form.is_want_news.data = fl.current_user.is_want_news
    return render_template(
        'update_profile.html',
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
    keyword = Keyword.query.filter_by(name=keyword_name).one_or_none()
    if keyword:
        return render_template('keyword.html', keyword=keyword)
    else:
        return _redirect_to_user(u'Keyword {} does not exist.'.format(
            keyword_name), is_error=True)


@app.route("/problems")
def problems():
    if fl.current_user.is_authenticated:
        db_tools.add_user_interaction(
            interaction='looking at problems', user=fl.current_user)
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
        if fl.current_user.is_authenticated:
            db_tools.add_user_interaction(
                interaction='looking at problem', user=fl.current_user,
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
            problem_name), is_error=True)


@app.route("/event_plots/<event_name>")
@fl.login_required
def event_plots(event_name):
    from bokeh.embed import components
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
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
    flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return redirect(url_for('problems'))


def _redirect_to_sandbox(event, message_str, is_error=True, category=None):
    flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return redirect(u'/events/{}/sandbox'.format(event.name))


def _redirect_to_credit(submission_hash, message_str, is_error=True,
                        category=None):
    flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return redirect(u'/credit/{}'.format(submission_hash))


@app.route("/events/<event_name>/sign_up")
@fl.login_required
def sign_up_for_event(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
    db_tools.add_user_interaction(
        interaction='signing up at event', user=fl.current_user, event=event)

    db_tools.ask_sign_up_team(event.name, fl.current_user.name)
    if event.is_controled_signup:
        db_tools.send_sign_up_request_mail(event, fl.current_user)
        return _redirect_to_user(
            "Sign-up request is sent to event admins.", is_error=False,
            category='Request sent')
    else:
        db_tools.sign_up_team(event.name, fl.current_user.name)
        return _redirect_to_sandbox(
            event, u'{} is signed up for {}.'.format(
                fl.current_user.firstname, event),
            is_error=False, category='Successful sign-up')


@app.route("/events/<event_name>/sign_up/<user_name>")
@fl.login_required
def approve_sign_up_for_event(event_name, user_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    user = User.query.filter_by(name=user_name).one_or_none()
    if not db_tools.is_admin(event, fl.current_user):
        return _redirect_to_user(
            u'Sorry {}, you do not have admin rights'.format(
                fl.current_user.firstname), is_error=True)
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
    if not fl.current_user.access_level == 'admin':
        return _redirect_to_user(
            u'Sorry {}, you do not have admin rights'.format(
                fl.current_user.firstname), is_error=True)
    if request.method == 'GET':
        asked_users = User.query.filter_by(access_level='asked')
        asked_sign_up = EventTeam.query.filter_by(approved=False)
        return render_template('approve.html', asked_users=asked_users,
                               asked_sign_up=asked_sign_up, admin=True)
    elif request.method == 'POST':
        users_to_be_approved = request.form.getlist('approve_users')
        event_teams_to_be_approved = request.form.getlist(
            'approve_event_teams')
        message = "Approved users:\n"
        for asked_user in users_to_be_approved:
            db_tools.approve_user(asked_user)
            message += "%s\n" % asked_user
        message += "Approved event_team:\n"
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
    if not fl.current_user.access_level == 'admin':
        return _redirect_to_user(
            u'Sorry {}, you do not have admin rights'.format(
                fl.current_user.firstname), is_error=True)
    if not user:
        return _redirect_to_user(
            u'Oups, no user {}.'.format(user_name), is_error=True)
    db_tools.approve_user(user.name)
    return _redirect_to_user(
        u'{} is signed up.'.format(user),
        is_error=False, category='Successful sign-up')


@app.route("/events/<event_name>")
@fl.login_required
def user_event(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        if fl.current_user.is_authenticated:
            return _redirect_to_user(u'{}: no event named "{}"'.format(
                fl.current_user.firstname, event_name))
        else:
            return _redirect_to_user(u'no event named "{}"'.format(
                event_name))
    if event:
        if fl.current_user.is_authenticated:
            db_tools.add_user_interaction(
                interaction='looking at event', user=fl.current_user,
                event=event)
        else:
            db_tools.add_user_interaction(
                interaction='looking at event', event=event)
        description_f_name = os.path.join(
            config.ramp_kits_path, event.problem.name,
            '{}_starting_kit.html'.format(event.problem.name))
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        admin = check_admin(fl.current_user, event)
        if fl.current_user.is_anonymous:
            approved = False
            asked = False
        else:
            approved = db_tools.is_user_signed_up(
                event_name, fl.current_user.name)
            asked = db_tools.is_user_asked_sign_up(
                event.name, fl.current_user.name)
        return render_template('event.html',
                               description=description,
                               event=event,
                               admin=admin,
                               approved=approved,
                               asked=asked)
    else:
        return _redirect_to_user(u'Event {} does not exist.'.format(
            event_name), is_error=True)


@app.route("/events/<event_name>/my_submissions")
@fl.login_required
def my_submissions(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
    db_tools.add_user_interaction(
        interaction='looking at my_submissions',
        user=fl.current_user, event=event)
    if not db_tools.is_open_code(event, fl.current_user):
        error_str = u'No access to my submissions for event {}. '\
            u'If you have already signed up, please wait for approval.'.\
            format(event.name)
        return _redirect_to_user(error_str)

    # Doesn't work if team mergers are allowed
    team = Team.query.filter_by(name=fl.current_user.name).one()
    event_team = EventTeam.query.filter_by(
        event=event, team=team).one_or_none()
    leaderboard_html = event_team.leaderboard_html
    failed_leaderboard_html = event_team.failed_leaderboard_html
    new_leaderboard_html = event_team.new_leaderboard_html
    admin = check_admin(fl.current_user, event)
    if event.official_score_type.is_lower_the_better:
        sorting_direction = 'asc'
    else:
        sorting_direction = 'desc'
    return render_template('leaderboard.html',
                           leaderboard_title='Trained submissions',
                           leaderboard=leaderboard_html,
                           failed_leaderboard=failed_leaderboard_html,
                           new_leaderboard=new_leaderboard_html,
                           sorting_column_index=4,
                           sorting_direction=sorting_direction,
                           event=event,
                           admin=admin)


@app.route("/events/<event_name>/leaderboard")
@fl.login_required
def leaderboard(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
    db_tools.add_user_interaction(
        interaction='looking at leaderboard',
        user=fl.current_user, event=event)

    if db_tools.is_open_leaderboard(event, fl.current_user):
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
        sorting_column_index=4,
        sorting_direction=sorting_direction,
        event=event
    )

    if fl.current_user.access_level == 'admin' or\
            db_tools.is_admin(event, fl.current_user):
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

    return template


@app.route("/events/<event_name>/competition_leaderboard")
@fl.login_required
def competition_leaderboard(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
    db_tools.add_user_interaction(
        interaction='looking at leaderboard',
        user=fl.current_user, event=event)
    admin = check_admin(fl.current_user, event)

    leaderboard_html = event.public_competition_leaderboard_html

    leaderboard_kwargs = dict(
        leaderboard=leaderboard_html,
        leaderboard_title='Leaderboard',
        sorting_column_index=0,
        sorting_direction='asc',
        event=event,
        admin=admin
    )

    return render_template('leaderboard.html', **leaderboard_kwargs)


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
    if submission is None or\
            not db_tools.is_open_code(
                submission.event_team.event, fl.current_user, submission):
        error_str = u'Missing submission: {}'.format(submission_hash)
        return _redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    workflow_element_name = f_name.split('.')[0]
    workflow_element = WorkflowElement.query.filter_by(
        name=workflow_element_name, workflow=event.workflow).one_or_none()
    if workflow_element is None:
        error_str = u'{} is not a valid workflow element by {} '.\
            format(workflow_element_name, fl.current_user.name)
        error_str += u'in {}/{}/{}/{}'.format(event, team, submission, f_name)
        return _redirect_to_user(error_str)
    submission_file = SubmissionFile.query.filter_by(
        submission=submission, workflow_element=workflow_element).one_or_none()
    if submission_file is None:
        error_str = u'No submission file by {} in {}/{}/{}/{}'.format(
            fl.current_user.name, event, team, submission, f_name)
        return _redirect_to_user(error_str)

    # superfluous, perhaps when we'll have different extensions?
    f_name = submission_file.f_name

    submission_abspath = os.path.abspath(submission.path)
    if not os.path.exists(submission_abspath):
        error_str = u'{} does not exist by {} in {}/{}/{}/{}'.format(
            submission_abspath, fl.current_user.name, event, team, submission,
            f_name)
        return _redirect_to_user(error_str)

    db_tools.add_user_interaction(
        interaction='looking at submission', user=fl.current_user, event=event,
        submission=submission, submission_file=submission_file)

    logger.info(u'{} is looking at {}/{}/{}/{}'.format(
        fl.current_user.name, event, team, submission, f_name))

    # Downloading file if it is not editable (e.g., external_data.csv)
    if not workflow_element.is_editable:
        # archive_filename = f_name  + '.zip'
        # with changedir(submission_abspath):
        #    with ZipFile(archive_filename, 'w') as archive:
        #        archive.write(f_name)
        db_tools.add_user_interaction(
            interaction='download', user=fl.current_user, event=event,
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
        sandbox_submission = db_tools.get_sandbox(event, fl.current_user)
        for f_name in import_form.selected_f_names.data:
            logger.info(u'{} is importing {}/{}/{}/{}'.format(
                fl.current_user.name, event, team, submission, f_name))

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
                interaction='copy', user=fl.current_user, event=event,
                submission=submission, submission_file=submission_file)

        return redirect(u'/events/{}/sandbox'.format(event.name))

    with open(os.path.join(submission.path, f_name)) as f:
        code = f.read()
    admin = check_admin(fl.current_user, event)
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
    submission = Submission.query.filter_by(
        hash_=submission_hash).one_or_none()
    if submission is None or\
            not db_tools.is_open_code(
                submission.event_team.event, fl.current_user, submission):
        error_str = u'Missing submission: {}'.format(submission_hash)
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


@app.route("/toggle_competition/<submission_hash>")
@fl.login_required
def toggle_competition(submission_hash):
    submission = Submission.query.filter_by(
        hash_=submission_hash).one_or_none()
    logger.info(submission)
    if submission is None or\
            submission.event_team.team.admin != fl.current_user:
        error_str = u'Missing submission: {}'.format(submission_hash)
        return _redirect_to_user(error_str)
    submission.is_in_competition = not submission.is_in_competition
    db.session.commit()
    db_tools.update_leaderboards(submission.event_team.event.name)
    return redirect(
        u'/{}/{}'.format(submission_hash, submission.files[0].f_name))


@app.route("/events/<event_name>/sandbox", methods=['GET', 'POST'])
@fl.login_required
def sandbox(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(
            u'{}: no access or no event named "{}"'.format(
                fl.current_user.firstname, event_name))
    if not db_tools.is_open_code(event, fl.current_user):
        error_str = u'No access to sandbox for event {}. '\
            u'If you have already signed up, please wait for approval.'.\
            format(event.name)
        return _redirect_to_user(error_str)

    sandbox_submission = db_tools.get_sandbox(event, fl.current_user)
    event_team = db_tools.get_active_user_event_team(event, fl.current_user)

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
                        interaction='save', user=fl.current_user, event=event,
                        submission_file=submission_file,
                        diff=diff, similarity=similarity)
        except Exception as e:
            return _redirect_to_sandbox(event, u'Error: {}'.format(e))
        return _redirect_to_sandbox(
            event, u'{} saved submission files for {}.'.format(
                fl.current_user.name, event_team, event),
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
            fl.current_user.name, upload_f_name, event))

        if submission_file.is_editable:
            new_code = submission_file.get_code()
            diff = '\n'.join(difflib.unified_diff(
                old_code.splitlines(), new_code.splitlines()))
            similarity = difflib.SequenceMatcher(
                a=old_code, b=new_code).ratio()
            db_tools.add_user_interaction(
                interaction='upload', user=fl.current_user, event=event,
                submission_file=submission_file,
                diff=diff, similarity=similarity)
        else:
            db_tools.add_user_interaction(
                interaction='upload', user=fl.current_user, event=event,
                submission_file=submission_file)

        return redirect(request.referrer)
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
            fl.current_user.name, new_submission.name, event_team))
        if event.is_send_submitted_mails:
            try:
                db_tools.send_submission_mails(
                    fl.current_user, new_submission, event_team)
            except Exception as e:
                error_str = u'mail was not sent {} '.format(
                    fl.current_user.name)
                error_str += u'submitted {} for {}\n{}.'.format(
                    new_submission.name, event_team, e)
                logger.error(error_str)
        flash(u'{} submitted {} for {}.'.format(
            fl.current_user.firstname, new_submission.name, event_team),
            category='Submission')

        db_tools.add_user_interaction(
            interaction='submit', user=fl.current_user, event=event,
            submission=new_submission)

        return redirect(u'/credit/{}'.format(new_submission.hash_))
    admin = check_admin(fl.current_user, event)
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
    if submission is None or\
            not db_tools.is_open_code(
                submission.event_team.event, fl.current_user, submission):
        error_str = u'Missing submission: {}'.format(submission_hash)
        return _redirect_to_user(error_str)
    event_team = submission.event_team
    event = event_team.event
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
    # new = True
    for source_submission in source_submissions:
        s_field = get_s_field(source_submission)
        submission_similaritys = SubmissionSimilarity.query.filter_by(
            type='target_credit', user=fl.current_user,
            source_submission=source_submission,
            target_submission=submission).all()
        if not submission_similaritys:
            credit = 0
        else:
            # new = False
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
            logger.info(sum_credit)
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
                type='target_credit', user=fl.current_user,
                source_submission=source_submission,
                target_submission=submission).all()
            # if submission_similarity is not empty, we need to
            # add zero to cancel previous credits explicitly
            if similarity > 0 or submission_similarity:
                submission_similarity = SubmissionSimilarity(
                    type='target_credit', user=fl.current_user,
                    source_submission=source_submission,
                    target_submission=submission,
                    similarity=similarity,
                    timestamp=datetime.datetime.utcnow())
                db.session.add(submission_similarity)
        db.session.commit()

        db_tools.add_user_interaction(
            interaction='giving credit', user=fl.current_user, event=event,
            submission=submission)

        return redirect(u'/events/{}/sandbox'.format(event.name))

    admin = check_admin(fl.current_user, event)
    return render_template('credit.html',
                           submission=submission,
                           source_submissions=source_submissions,
                           credit_form=credit_form,
                           event=event,
                           admin=admin)


@app.route("/logout")
@fl.login_required
def logout():
    user = fl.current_user
    db_tools.add_user_interaction(interaction='logout', user=user)
    session['logged_in'] = False
    user.is_authenticated = False
    db.session.commit()
    logger.info(u'{} is logged out'.format(user))
    fl.logout_user()

    return redirect(url_for('login'))


@app.route("/events/<event_name>/private_leaderboard")
@fl.login_required
def private_leaderboard(event_name):
    # start = time.time()

    if not fl.current_user.is_authenticated:
        return redirect(url_for('login'))
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
    if (not db_tools.is_admin(event, fl.current_user) and
        (event.closing_timestamp is None or
            event.closing_timestamp > datetime.datetime.utcnow())):
        return redirect(url_for('problems'))

    db_tools.add_user_interaction(
        interaction='looking at private leaderboard',
        user=fl.current_user, event=event)
    leaderboard_html = event.private_leaderboard_html
    admin = check_admin(fl.current_user, event)
    if event.official_score_type.is_lower_the_better:
        sorting_direction = 'asc'
    else:
        sorting_direction = 'desc'

    template = render_template(
        'leaderboard.html',
        leaderboard_title='Leaderboard',
        leaderboard=leaderboard_html,
        sorting_column_index=5,
        sorting_direction=sorting_direction,
        event=event,
        private=True,
        admin=admin
    )

    # logger.info(u'private leaderboard takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    return template


@app.route("/events/<event_name>/private_competition_leaderboard")
@fl.login_required
def private_competition_leaderboard(event_name):
    if not fl.current_user.is_authenticated:
        return redirect(url_for('login'))
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
    if (not db_tools.is_admin(event, fl.current_user) and
        (event.closing_timestamp is None or
            event.closing_timestamp > datetime.datetime.utcnow())):
        return redirect(url_for('problems'))

    db_tools.add_user_interaction(
        interaction='looking at private leaderboard',
        user=fl.current_user, event=event)

    admin = check_admin(fl.current_user, event)
    leaderboard_html = event.private_competition_leaderboard_html

    leaderboard_kwargs = dict(
        leaderboard=leaderboard_html,
        leaderboard_title='Leaderboard',
        sorting_column_index=0,
        sorting_direction='asc',
        event=event,
        admin=admin
    )

    return render_template('leaderboard.html', **leaderboard_kwargs)


@app.route("/events/<event_name>/update", methods=['GET', 'POST'])
@fl.login_required
def update_event(event_name):
    if not fl.current_user.is_authenticated:
        return redirect(url_for('login'))
    event = Event.query.filter_by(name=event_name).one_or_none()
    if not db_tools.is_public_event(event, fl.current_user):
        return _redirect_to_user(u'{}: no event named "{}"'.format(
            fl.current_user.firstname, event_name))
    if not db_tools.is_admin(event, fl.current_user):
        return redirect(url_for('problems'))
    logger.info(u'{} is updating event {}'.format(
        fl.current_user.name, event.name))
    admin = check_admin(fl.current_user, event)
    # We assume here that event name has the syntax <problem_name>_<suffix>
    suffix = event.name[len(event.problem.name) + 1:]

    h = event.min_duration_between_submissions // 3600
    m = event.min_duration_between_submissions // 60 % 60
    s = event.min_duration_between_submissions % 60
    form = EventUpdateProfileForm(
        suffix=suffix, title=event.title,
        is_send_trained_mails=event.is_send_trained_mails,
        is_send_submitted_mails=event.is_send_submitted_mails,
        is_public=event.is_public,
        is_controled_signup=event.is_controled_signup,
        is_competitive=event.is_competitive,
        min_duration_between_submissions_hour=h,
        min_duration_between_submissions_minute=m,
        min_duration_between_submissions_second=s,
        opening_timestamp=event.opening_timestamp,
        closing_timestamp=event.closing_timestamp,
        public_opening_timestamp=event.public_opening_timestamp,
    )
    if form.validate_on_submit():
        try:
            if form.suffix.data == '':
                event.name = event.problem.name
            else:
                event.name = event.problem.name + '_' + form.suffix.data
            event.title = form.title.data
            event.is_send_trained_mails = form.is_send_trained_mails.data
            event.is_send_submitted_mails = form.is_send_submitted_mails.data
            event.is_public = form.is_public.data
            event.is_controled_signup = form.is_controled_signup.data
            event.is_competitive = form.is_competitive.data
            event.min_duration_between_submissions = (
                form.min_duration_between_submissions_hour.data * 3600 +
                form.min_duration_between_submissions_minute.data * 60 +
                form.min_duration_between_submissions_second.data)
            event.opening_timestamp = form.opening_timestamp.data
            event.closing_timestamp = form.closing_timestamp.data
            event.public_opening_timestamp = form.public_opening_timestamp.data
            db.session.commit()

        except IntegrityError as e:
            db.session.rollback()
            message = ''
            existing_event = Event.query.filter_by(
                name=event.name).one_or_none()
            if existing_event is not None:
                message += 'event name is already in use'
            # # try:
            # #     User.query.filter_by(email=email).one()
            # #     if len(message) > 0:
            # #         message += ' and '
            # #     message += 'email is already in use'
            # except NoResultFound:
            #     pass
            if len(message) > 0:
                e = NameClashError(message)
            flash(u'{}'.format(e), category='Update event error')
            return redirect(url_for('update_event', event_name=event.name))

        return redirect(url_for('problems'))

    return render_template(
        'update_event.html',
        form=form,
        event=event,
        admin=admin,
    )


@app.route("/problems/<problem_name>/ask_for_event", methods=['GET', 'POST'])
@fl.login_required
def ask_for_event(problem_name):
    if not fl.current_user.is_authenticated:
        return redirect(url_for('login'))
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    if problem is None:
        return _redirect_to_user(u'{}: no problem named "{}"'.format(
            fl.current_user.firstname, problem_name))
    logger.info(u'{} is asking for event on {}'.format(
        fl.current_user.name, problem.name))
    # We assume here that event name has the syntax <problem_name>_<suffix>
    form = AskForEventForm(
        min_duration_between_submissions_hour=8,
        min_duration_between_submissions_minute=0,
        min_duration_between_submissions_second=0,
    )
    if form.validate_on_submit():
        try:
            event_name = problem.name + '_' + form.suffix.data
            event_title = form.title.data
            event = db_tools.add_event(problem_name, event_name, event_title)
            event.min_duration_between_submissions = (
                form.min_duration_between_submissions_hour.data * 3600 +
                form.min_duration_between_submissions_minute.data * 60 +
                form.min_duration_between_submissions_second.data)
            event.opening_timestamp = form.opening_date.data
            event.closing_timestamp = form.closing_date.data
            flash(
                'Thank you. Your request has been sent to RAMP ' +
                'administrators.', category='Event request')
            db_tools.send_ask_for_event_mails(
                fl.current_user, event, form.n_students.data)
        except IntegrityError as e:
            db.session.rollback()
            message = ''
            existing_event = Event.query.filter_by(
                name=event_name).one_or_none()
            if existing_event is not None:
                message += 'event name is already in use'
            if len(message) > 0:
                e = NameClashError(message)
            flash(u'{}'.format(e), category='Ask for event error')
            return redirect(url_for(
                'ask_for_event', problem_name=problem_name))

        return redirect(url_for('problems'))

    return render_template(
        'ask_for_event.html',
        form=form,
        problem=problem,
    )


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
            fl.current_user.name, submission_hash)
        return _redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    # TODO: check if event == submission.event_team.event

    db_tools.add_user_interaction(
        interaction='looking at error', user=fl.current_user, event=event,
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
    if not fl.current_user.is_authenticated\
            or fl.current_user.access_level != 'admin':
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
    if not fl.current_user.is_authenticated\
            or fl.current_user.access_level != 'admin':
        return redirect(url_for('login'))
    user_interaction = UserInteraction.query.filter_by(id=id).one()
    return render_template(
        'submission_file_diff.html',
        diff=user_interaction.submission_file_diff
    )


@app.after_request
def after_request(response):
    for query in fs.get_debug_queries():
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

    if fl.current_user.access_level == 'admin' or\
            db_tools.is_admin(event, fl.current_user):
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
            format(fl.current_user.firstname, event_name))


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
