import os
# import sys
import shutil
import difflib
import logging
import os.path
import smtplib
import tempfile
import datetime

from git import Repo
from zipfile import ZipFile
from flask import request, redirect, url_for, render_template,\
    send_from_directory, flash, session, g
from flask.ext.login import current_user, login_required, AnonymousUserMixin
from sqlalchemy.orm.exc import NoResultFound
from werkzeug import secure_filename
from wtforms import StringField
from wtforms.widgets import TextArea

import flask
import flask.ext.login as fl
from databoard import app, login_manager
from databoard.generic import changedir
from databoard.config import repos_path
import databoard.config as config
import databoard.db_tools as db_tools
from databoard.model import User, Team, Submission, WorkflowElement,\
    SubmissionFile, UserInteraction,\
    DuplicateSubmissionError, TooEarlySubmissionError,\
    MissingExtensionError
from databoard.forms import LoginForm, CodeForm, SubmitForm, ImportForm,\
    UploadForm

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
        fl.login_user(user, remember=True)  # , remember=form.remember_me.data)
        session['logged_in'] = True
        logger.info('{} is logged in'.format(current_user))
        db_tools.add_user_interaction(user=current_user, interaction='login')
        # next = flask.request.args.get('next')
        # next_is_valid should check if the user has valid
        # permission to access the `next` url
        #if not fl.next_is_valid(next):
        #    return flask.abort(400)

        return flask.redirect(flask.url_for('user'))
    return render_template(
        'login.html',
        ramp_title=config.config_object.specific.ramp_title,
        form=form,
        opening_date_time=db_tools.date_time_format(config.opening_timestamp),
        public_opening_date_time=db_tools.date_time_format(
            config.public_opening_timestamp),
        min_duration_between_submissions='{} minutes'.format(
            config.min_duration_between_submissions / 60)
    )


def send_submission_mails(user, submission, team):
    #  later can be joined to the ramp admins
    recipient_list = config.ADMIN_MAILS
    gmail_user = config.MAIL_USERNAME
    gmail_pwd = config.MAIL_PASSWORD
    smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    subject = 'fab train_test:t={},s={}'.format(team.name, submission.name)
    header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
        recipient_list, gmail_user, subject)
    body = 'user = {}\nramp = {}\nserver = {}\nsubmission dir = {}\n'.format(
        user,
        config.config_object.ramp_name,
        config.config_object.get_deployment_target(mode='train'),
        submission.path)
    for recipient in recipient_list:
        smtpserver.sendmail(gmail_user, recipient, header + body)


@app.route("/sandbox", methods=['GET', 'POST'])
@fl.login_required
def sandbox():
    if current_user.access_level != 'admin' and\
            datetime.datetime.utcnow() < config.opening_timestamp:
        flask.flash('Submission will open at (UTC) {}'.format(
            db_tools.date_time_format(config.opening_timestamp)))
        return flask.redirect(flask.url_for('login'))

    sandbox_submission = db_tools.get_sandbox(current_user)
    team = db_tools.get_active_user_team(current_user)

    # The amount of python magic we have to do for rendering a variable number
    # of textareas, named and populated at run time, is mind boggling. 
    # We first create named
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

    submit_form = SubmitForm(submission_name=team.last_submission_name)
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
                        user=current_user, interaction='save',
                        submission_file=submission_file, 
                        diff=diff, similarity=similarity)
        except Exception as e:
            flask.flash('Error: {}'.format(e))
            return flask.redirect(flask.url_for('sandbox'))
        logger.info('{} saved files'.format(current_user))
        flask.flash('{} saved submission files for team {}.'.format(
            current_user, db_tools.get_active_user_team(current_user)),
            category='File saved')
        return flask.redirect(flask.url_for('sandbox'))

    if upload_form.validate_on_submit() and upload_form.file.data:
        upload_f_name = secure_filename(upload_form.file.data.filename)
        upload_name = upload_f_name.split('.')[0]
        upload_workflow_element = WorkflowElement.query.filter_by(
            name=upload_name).one_or_none()
        if upload_workflow_element is None:
            flask.flash('{} is not in the file list.'.format(upload_f_name))
            return flask.redirect(flask.url_for('sandbox'))

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
            flask.flash('File is too big: {} exceeds max size {}'.format(
                file_length, upload_workflow_element.max_size))
            return flask.redirect(flask.url_for('sandbox'))
        dst = os.path.join(sandbox_submission.path, upload_f_name)
        shutil.copy2(tmp_f_name, dst)
        logger.info('{} uploaded {}'.format(current_user, upload_f_name))

        if submission_file.is_editable:
            new_code = submission_file.get_code()
            diff = '\n'.join(difflib.unified_diff(
                old_code.splitlines(), new_code.splitlines()))
            similarity = difflib.SequenceMatcher(
                a=old_code, b=new_code).ratio()
            db_tools.add_user_interaction(
                user=current_user, interaction='upload',
                submission_file=submission_file,
                diff=diff, similarity=similarity)
        else:
            db_tools.add_user_interaction(
                user=current_user, interaction='upload')

        return flask.redirect(request.referrer)
        # TODO: handle different extensions for the same workflow element
        # ie: now we let upload eg external_data.bla, and only fail at 
        # submission, without giving a message

    if submit_form.validate_on_submit() and submit_form.submission_name.data:
        new_submission_name = submit_form.submission_name.data
        if len(new_submission_name) < 4 or len(new_submission_name) > 20:
            flask.flash(
                'Submission name should have length between 4 and 20 ' +
                'characters.')
            return flask.redirect(flask.url_for('sandbox'))

        try:
            new_submission_name.encode('ascii')
        except Exception as e:
            flask.flash('Error: {}'.format(e))
            return flask.redirect(flask.url_for('sandbox'))

        try:
            new_submission = db_tools.make_submission_and_copy_files(
                team.name, new_submission_name, sandbox_submission.path)
        except DuplicateSubmissionError:
            flask.flash(
                'Submission {} already exists. Please change the name.'.format(
                    new_submission_name))
            return flask.redirect(flask.url_for('sandbox'))
        except MissingExtensionError as e:
            flask.flash('Missing extension, {}'.format(e.value))
            return flask.redirect(flask.url_for('sandbox'))
        except TooEarlySubmissionError as e:
            flask.flash(e.value)
            return flask.redirect(flask.url_for('sandbox'))

        logger.info('{} submitted {} for team {}.'.format(
            current_user, new_submission.name, team))
        send_submission_mails(current_user, new_submission, team)
        flask.flash('{} submitted {} for team {}.'.format(
            current_user, new_submission.name, team),
            category='Submission')

        db_tools.add_user_interaction(
            user=current_user, interaction='submit',
            submission=new_submission)

        return flask.redirect(flask.url_for('user'))
    return render_template('sandbox.html',
                           ramp_title=config.config_object.specific.ramp_title,
                           submission_names=sandbox_submission.f_names,
                           code_form=code_form,
                           submit_form=submit_form,
                           upload_form=upload_form)


@app.route("/user", methods=['GET', 'POST'])
@fl.login_required
def user():
    db_tools.add_user_interaction(
        user=current_user, interaction='looking at user')

    leaderbord_html = db_tools.get_public_leaderboard(user=current_user)
    failed_submissions_html = db_tools.get_failed_submissions(
        user=current_user)
    new_submissions_html = db_tools.get_new_submissions(
        user=current_user)
    return render_template('leaderboard.html',
                           leaderboard_title='Trained submissions',
                           leaderboard=leaderbord_html,
                           failed_submissions=failed_submissions_html,
                           new_submissions=new_submissions_html,
                           ramp_title=config.config_object.specific.ramp_title)


@app.route("/logout")
@fl.login_required
def logout():
    db_tools.add_user_interaction(
        user=current_user, interaction='logout')
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
def leaderboard():
    if current_user.is_authenticated:
        db_tools.add_user_interaction(
            user=current_user, interaction='looking at leaderboard')
        if current_user.access_level == 'admin':
            leaderbord_html = db_tools.get_public_leaderboard()
            failed_submissions_html = db_tools.get_failed_submissions()
            new_submissions_html = db_tools.get_new_submissions()
            return render_template(
                'leaderboard.html',
                leaderboard_title='Leaderboard',
                leaderboard=leaderbord_html,
                failed_submissions=failed_submissions_html,
                new_submissions=new_submissions_html,
                ramp_title=config.config_object.specific.ramp_title)
        else:
            if config.public_opening_timestamp < datetime.datetime.utcnow():
                leaderbord_html = db_tools.get_public_leaderboard()
                return render_template(
                    'leaderboard.html',
                    leaderboard_title='Leaderboard',
                    leaderboard=leaderbord_html,
                    ramp_title=config.config_object.specific.ramp_title)
            else:
                leaderbord_html = db_tools.get_public_leaderboard(
                    is_open_code=False)
                return render_template(
                    'leaderboard.html',
                    leaderboard_title='Leaderboard',
                    leaderboard=leaderbord_html,
                    opening_date_time=db_tools.date_time_format(
                        config.public_opening_timestamp),
                    ramp_title=config.config_object.specific.ramp_title)
    else:
        leaderbord_html = db_tools.get_public_leaderboard(is_open_code=False)
        return render_template(
            'leaderboard.html',
            leaderboard_title='Leaderboard',
            leaderboard=leaderbord_html,
            ramp_title=config.config_object.specific.ramp_title)


@app.route("/private_leaderboard")
@fl.login_required
def private_leaderboard():
    if ((not current_user.is_authenticated
         or current_user.access_level != 'admin')
        and (config.closing_timestamp is None or
             config.closing_timestamp > datetime.datetime.utcnow())):
        return redirect(url_for('leaderboard'))
    leaderbord_html = db_tools.get_private_leaderboard()
    return render_template(
        'leaderboard.html',
        leaderboard_title='Private leaderboard',
        leaderboard=leaderbord_html,
        ramp_title=config.config_object.specific.ramp_title)


@app.route("/user_interactions")
@fl.login_required
def user_interactions():
    if not current_user.is_authenticated\
            or current_user.access_level != 'admin':
        return redirect(url_for('leaderboard'))
    user_interactions_html = db_tools.get_user_interactions()
    return render_template(
        'user_interactions.html',
        user_interactions_title='User interactions',
        user_interactions=user_interactions_html,
        ramp_title=config.config_object.specific.ramp_title)


@app.route("/submissions/<team_name>/<summission_hash>/<f_name>",
           methods=['GET', 'POST'])
@fl.login_required
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

    team = Team.query.filter_by(name=team_name).one_or_none()
    submission = Submission.query.filter_by(
        team=team, hash_=summission_hash).one_or_none()
    name = f_name.split('.')[0]
    workflow_element = WorkflowElement.query.filter_by(name=name).one_or_none()
    if team is None or submission is None or workflow_element is None:
        logger.error('{}/{}/{}'.format(team, submission, workflow_element))
        return redirect(url_for('leaderboard'))

    submission_file = SubmissionFile.query.filter_by(
        submission=submission, workflow_element=workflow_element).one_or_none()
    if submission_file is None:
        logger.error('No submission file')
        return redirect(url_for('leaderboard'))


    # superfluous, perhaps when we'll have different extensions?
    f_name = submission_file.f_name

    submission_abspath = os.path.abspath(submission.path)
    if not os.path.exists(submission_abspath):
        logger.error('{} does not exist'.format(submission_abspath))
        return redirect(url_for('leaderboard'))

    db_tools.add_user_interaction(
        user=current_user, interaction='looking at submission',
        submission=submission, submission_file=submission_file)

    logger.info('{} is looking at {}/{}/{}'.format(
        current_user, team, submission, f_name))

    # Downloading file if it is not editable (e.g., external_data.csv)
    if not workflow_element.is_editable:
        #archive_filename = f_name  + '.zip'
        #with changedir(submission_abspath):
        #    with ZipFile(archive_filename, 'w') as archive:
        #        archive.write(f_name)
        db_tools.add_user_interaction(
            user=current_user, interaction='download',
            submission=submission, submission_file=submission_file)

        return send_from_directory(
            submission_abspath, f_name, as_attachment=True,
            attachment_filename='{}_{}_{}'.format(
                team_name, submission.hash_[:6], f_name),
            mimetype='application/octet-stream')

    # Importing selected files into sandbox
    choices = [(f, f) for f in submission.f_names]
    import_form = ImportForm()
    import_form.selected_f_names.choices = choices
    if import_form.validate_on_submit():
        sandbox_submission = db_tools.get_sandbox(current_user)
        for f_name in import_form.selected_f_names.data:
            logger.info('{} is importing {}/{}/{}'.format(
                current_user, team, submission, f_name))

            # TODO: deal with different extensions of the same file
            src = os.path.join(submission.path, f_name)
            dst = os.path.join(sandbox_submission.path, f_name)
            shutil.copy2(src, dst)  # copying also metadata
            logger.info('Copying {} to {}'.format(src, dst))

            workflow_element = WorkflowElement.query.filter_by(
                name=f_name.split('.')[0]).one()
            submission_file = SubmissionFile.query.filter_by(
                submission=submission,
                workflow_element=workflow_element).one()
            db_tools.add_user_interaction(
                user=current_user, interaction='copy',
                submission=submission, submission_file=submission_file)

        return flask.redirect(flask.url_for('sandbox'))

    with open(os.path.join(submission.path, f_name)) as f:
        code = f.read()
    return render_template(
        'submission.html',
        code=code,
        submission_f_names=submission.f_names,
        f_name=f_name,
        submission_name=submission.name,
        team_name=team.name,
        ramp_title=specific.ramp_title,
        import_form=import_form)


@app.route("/submissions/<team_name>/<summission_hash>/error.txt")
@fl.login_required
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

    submission_url = request.path.rstrip('/') + '/raw'

    db_tools.add_user_interaction(
        user=current_user, interaction='looking at error',
        submission = submission)

    return render_template(
        'submission_error.html',
        error_msg=submission.error_msg,
        submission_state=submission.state,
        submission_url=submission_url,
        submission_f_names=submission.f_names,
        submission_name=submission.name,
        team_name=team.name,
        ramp_title=specific.ramp_title)


@app.route("/submissions/diff_bef24208a45043059/<id>")
@fl.login_required
def submission_file_diff(id):
    if not current_user.is_authenticated\
            or current_user.access_level != 'admin':
        return redirect(url_for('leaderboard'))
    user_interaction = UserInteraction.query.filter_by(id=id).one()
    return render_template(
        'submission_file_diff.html',
        diff=user_interaction.submission_file_diff,
        ramp_title=config.config_object.specific.ramp_title)


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
