import os
# import sys
import shutil
import difflib
import logging
import os.path
import tempfile
import datetime

from flask import request, redirect, url_for, render_template,\
    send_from_directory, session, g
from flask.ext.login import current_user
from sqlalchemy.orm.exc import NoResultFound
from werkzeug import secure_filename
from wtforms import StringField
from wtforms.widgets import TextArea

import flask
import flask.ext.login as fl
from databoard import app, login_manager
import databoard.db_tools as db_tools
from databoard.model import User, Submission, WorkflowElement,\
    Event, SubmissionFile, UserInteraction,\
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
        # if not fl.next_is_valid(next):
        #    return flask.abort(400)

        return flask.redirect(flask.url_for('user'))
    return render_template(
        'login.html',
        form=form,
    )


@app.route("/user")
@fl.login_required
def user():
    db_tools.add_user_interaction(
        user=current_user, interaction='looking at user')

    events = Event.query.all()
    event_urls_f_names = [(event.name, event.title) for event in events]
    return render_template('user.html',
                           event_urls_f_names=event_urls_f_names)


def _redirect_to_user(error_str):
    flask.flash(error_str)
    logger.error(error_str)
    return redirect(url_for('user'))


def _redirect_to_sandbox(event, message_str, is_error=True, category=None):
    flask.flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return flask.redirect('/events/{}/sandbox'.format(event.name))


@app.route("/events/<event_name>")
@fl.login_required
def user_event(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is None:
        return _redirect_to_user('{}: no event named "{}"'.format(
            current_user, event_name))
    db_tools.add_user_interaction(
        user=current_user, event=event, interaction='looking at event')

    return render_template('event.html',
                           event=event)


@app.route("/events/<event_name>/my_submissions")
@fl.login_required
def my_submissions(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is None:
        return _redirect_to_user('{}: no event named "{}"'.format(
            current_user, event_name))
    db_tools.add_user_interaction(
        user=current_user, event=event,
        interaction='looking at my_submissions')

    leaderbord_html = db_tools.get_public_leaderboard(
        event_name, current_user, user_name=current_user.name)
    failed_leaderboard_html = db_tools.get_failed_leaderboard(
        event_name, user_name=current_user.name)
    new_leaderboard_html = db_tools.get_new_leaderboard(
        event_name, user_name=current_user.name)
    return render_template('leaderboard.html',
                           leaderboard_title='Trained submissions',
                           leaderboard=leaderbord_html,
                           failed_leaderboard=failed_leaderboard_html,
                           new_leaderboard=new_leaderboard_html,
                           event=event)


@app.route("/events/<event_name>/leaderboard")
def leaderboard(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is None:
        return _redirect_to_user('{}: no event named "{}"'.format(
            current_user, event_name))
    db_tools.add_user_interaction(
        user=current_user, event=event,
        interaction='looking at leaderboard')
    leaderbord_html = db_tools.get_public_leaderboard(event_name, current_user)
    leaderboard_kwargs = dict(
        leaderboard=leaderbord_html,
        leaderboard_title='Leaderboard',
        event=event
    )

    if current_user.access_level == 'admin' or\
            db_tools.is_admin(event, current_user):
        failed_leaderboard_html = db_tools.get_failed_leaderboard(event_name)
        new_leaderboard_html = db_tools.get_new_leaderboard(event_name)
        return render_template(
            'leaderboard.html',
            failed_leaderboard=failed_leaderboard_html,
            new_leaderboard=new_leaderboard_html,
            **leaderboard_kwargs)
    else:
        return render_template(
            'leaderboard.html',
            **leaderboard_kwargs)


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
        error_str = 'Missing submission {}: {}/{}'.format(
            current_user, submission_hash, f_name)
        return _redirect_to_user(error_str)
    event = submission.event_team.event
    if not db_tools.is_open_code(event, current_user, submission):
        error_str = '{} has no right to look at {}/{}/{}'.format(
            current_user, event, submission, f_name)
        return _redirect_to_user(error_str)
    team = submission.event_team.team
    workflow_element_name = f_name.split('.')[0]
    workflow_element = WorkflowElement.query.filter_by(
        name=workflow_element_name).one_or_none()
    if workflow_element is None:
        error_str = '{} is not a valid workflow element by {} in {}/{}/{}/{}'.\
            format(workflow_element_name, current_user, event, team,
                   submission, f_name)
        return _redirect_to_user(error_str)
    submission_file = SubmissionFile.query.filter_by(
        submission=submission, workflow_element=workflow_element).one_or_none()
    if submission_file is None:
        error_str = 'No submission file by {} in {}/{}/{}/{}'.format(
            current_user, event, team, submission, f_name)
        return _redirect_to_user(error_str)

    # superfluous, perhaps when we'll have different extensions?
    f_name = submission_file.f_name

    submission_abspath = os.path.abspath(submission.path)
    if not os.path.exists(submission_abspath):
        error_str = '{} does not exist by {} in {}/{}/{}/{}'.format(
            submission_abspath, current_user, event, team, submission, f_name)
        return _redirect_to_user(error_str)

    db_tools.add_user_interaction(
        user=current_user, event=event, interaction='looking at submission',
        submission=submission, submission_file=submission_file)

    logger.info('{} is looking at {}/{}/{}/{}'.format(
        current_user, event, team, submission, f_name))

    # Downloading file if it is not editable (e.g., external_data.csv)
    if not workflow_element.is_editable:
        # archive_filename = f_name  + '.zip'
        # with changedir(submission_abspath):
        #    with ZipFile(archive_filename, 'w') as archive:
        #        archive.write(f_name)
        db_tools.add_user_interaction(
            user=current_user, event=event, interaction='download',
            submission=submission, submission_file=submission_file)

        return send_from_directory(
            submission_abspath, f_name, as_attachment=True,
            attachment_filename='{}_{}'.format(submission.hash_[:6], f_name),
            mimetype='application/octet-stream')

    # Importing selected files into sandbox
    choices = [(f, f) for f in submission.f_names]
    import_form = ImportForm()
    import_form.selected_f_names.choices = choices
    if import_form.validate_on_submit():
        sandbox_submission = db_tools.get_sandbox(event, current_user)
        for f_name in import_form.selected_f_names.data:
            logger.info('{} is importing {}/{}/{}/{}'.format(
                current_user, event, team, submission, f_name))

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
                user=current_user, event=event, interaction='copy',
                submission=submission, submission_file=submission_file)

        return flask.redirect('/events/{}/sandbox'.format(event.name))

    with open(os.path.join(submission.path, f_name)) as f:
        code = f.read()
    return render_template(
        'submission.html',
        code=code,
        submission_f_names=submission.f_names,
        f_name=f_name,
        submission_name=submission.name,
        team_name=team.name,
        import_form=import_form,
        event=event)


@app.route("/events/<event_name>/sandbox", methods=['GET', 'POST'])
@fl.login_required
def sandbox(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is None:
        return _redirect_to_user('{}: no event named "{}"'.format(
            current_user, event_name))
    if not db_tools.is_open_code(event, current_user):
        error_str = 'No sandbox of {} in {}'.format(current_user, event)
        return _redirect_to_user(error_str)

    sandbox_submission = db_tools.get_sandbox(event, current_user)
    event_team = db_tools.get_active_user_event_team(event, current_user)

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
                        user=current_user, event=event, interaction='save',
                        submission_file=submission_file,
                        diff=diff, similarity=similarity)
        except Exception as e:
            return _redirect_to_sandbox(event, 'Error: {}'.format(e))
        return _redirect_to_sandbox(
            event, '{} saved submission files for {}.'.format(
                current_user, event_team, event),
            is_error=False, category='File saved')

    if upload_form.validate_on_submit() and upload_form.file.data:
        upload_f_name = secure_filename(upload_form.file.data.filename)
        upload_name = upload_f_name.split('.')[0]
        upload_workflow_element = WorkflowElement.query.filter_by(
            name=upload_name).one_or_none()
        if upload_workflow_element is None:
            return _redirect_to_sandbox(
                event, '{} is not in the file list.'.format(upload_f_name))

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
                event, 'File is too big: {} exceeds max size {}'.format(
                    file_length, upload_workflow_element.max_size))
        dst = os.path.join(sandbox_submission.path, upload_f_name)
        shutil.copy2(tmp_f_name, dst)
        logger.info('{} uploaded {} in {}'.format(
            current_user, upload_f_name, event))

        if submission_file.is_editable:
            new_code = submission_file.get_code()
            diff = '\n'.join(difflib.unified_diff(
                old_code.splitlines(), new_code.splitlines()))
            similarity = difflib.SequenceMatcher(
                a=old_code, b=new_code).ratio()
            db_tools.add_user_interaction(
                user=current_user, event=event, interaction='upload',
                submission_file=submission_file,
                diff=diff, similarity=similarity)
        else:
            db_tools.add_user_interaction(
                user=current_user, event=event, interaction='upload',
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
            return _redirect_to_sandbox(event, 'Error: {}'.format(e))

        try:
            new_submission = db_tools.make_submission_and_copy_files(
                event.name, event_team.team.name, new_submission_name,
                sandbox_submission.path)
        except DuplicateSubmissionError:
            return _redirect_to_sandbox(
                event, 'Submission {} already exists. Please change the name.'
                .format(new_submission_name))
        except MissingExtensionError as e:
            return _redirect_to_sandbox(
                event, 'Missing extension, {}'.format(e.value))
        except TooEarlySubmissionError as e:
            return _redirect_to_sandbox(event, e.value)

        logger.info('{} submitted {} for {}.'.format(
            current_user, new_submission.name, event_team))
        if event.is_send_submitted_mails:
            db_tools.send_submission_mails(
                current_user, new_submission, event_team)
        flask.flash('{} submitted {} for {}.'.format(
            current_user, new_submission.name, event_team),
            category='Submission')

        db_tools.add_user_interaction(
            user=current_user, event=event, interaction='submit',
            submission=new_submission)

        return flask.redirect(flask.url_for('user'))
    return render_template('sandbox.html',
                           submission_names=sandbox_submission.f_names,
                           code_form=code_form,
                           submit_form=submit_form,
                           upload_form=upload_form,
                           event=event)


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


@app.route("/events/<event_name>/private_leaderboard")
@fl.login_required
def private_leaderboard(event_name):
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is None:
        return _redirect_to_user('{}: no event named "{}"'.format(
            current_user, event_name))
    if not current_user.is_authenticated:
        return redirect(url_for('leaderboard'))
    if (not db_tools.is_admin(event, current_user) and
        (event.closing_timestamp is None or
            event.closing_timestamp > datetime.datetime.utcnow())):
        return redirect('/events/{}/private_leaderboard'.format(event_name))

    db_tools.add_user_interaction(
        user=current_user, event=event,
        interaction='looking at private leaderboard')
    leaderbord_html = db_tools.get_private_leaderboard(event_name)
    return render_template(
        'leaderboard.html',
        leaderboard_title='Leaderboard',
        leaderboard=leaderbord_html,
        event=event
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
        error_str = 'Missing submission {}: {}'.format(
            current_user, submission_hash)
        return _redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    # TODO: check if event == submission.event_team.event

    db_tools.add_user_interaction(
        user=current_user, event=event, interaction='looking at error',
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
