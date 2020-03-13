import datetime
import difflib
import logging
import io
import os
import shutil
import tempfile
import time
import zipfile

from bokeh.embed import components

import flask_login

from flask import Blueprint
from flask import current_app as app
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory
from flask import send_file

from wtforms import StringField
from wtforms.widgets import TextArea

from werkzeug.utils import secure_filename

from ramp_database.model import Event
from ramp_database.model import Problem
from ramp_database.model import Submission
from ramp_database.model import SubmissionFile
from ramp_database.model import SubmissionSimilarity
from ramp_database.model import User
from ramp_database.model import WorkflowElement

from ramp_database.exceptions import DuplicateSubmissionError
from ramp_database.exceptions import MissingExtensionError
from ramp_database.exceptions import TooEarlySubmissionError

from ramp_database.tools.event import get_event
from ramp_database.tools.event import get_problem
from ramp_database.tools.frontend import is_admin
from ramp_database.tools.frontend import is_accessible_code
from ramp_database.tools.frontend import is_accessible_event
from ramp_database.tools.frontend import is_user_signed_up
from ramp_database.tools.frontend import is_user_sign_up_requested
from ramp_database.tools.leaderboard import update_leaderboards
from ramp_database.tools.submission import add_submission
from ramp_database.tools.submission import add_submission_similarity
from ramp_database.tools.submission import get_source_submissions
from ramp_database.tools.submission import get_submission_by_name
from ramp_database.tools.user import add_user_interaction
from ramp_database.tools.team import ask_sign_up_team
from ramp_database.tools.team import get_event_team_by_name
from ramp_database.tools.team import sign_up_team

from ramp_frontend import db

from ..forms import AskForEventForm
from ..forms import CodeForm
from ..forms import CreditForm
from ..forms import ImportForm
from ..forms import SubmitForm
from ..forms import UploadForm

from ..utils import body_formatter_user
from ..utils import send_mail

from .redirect import redirect_to_credit
from .redirect import redirect_to_sandbox
from .redirect import redirect_to_user

from .visualization import score_plot

mod = Blueprint('ramp', __name__)
logger = logging.getLogger('RAMP-FRONTEND')


@mod.route("/problems")
def problems():
    """Landing page showing all the RAMP problems."""
    user = (flask_login.current_user
            if flask_login.current_user.is_authenticated else None)
    admin = user.access_level == 'admin' if user is not None else False
    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session, interaction='looking at problems', user=user
        )
    problems = get_problem(db.session, None)

    for problem in problems:
        for event in problem.events:
            # check the state of the event
            now = datetime.datetime.now()
            start = event.opening_timestamp
            start_collab = event.public_opening_timestamp
            end = event.closing_timestamp
            if now < start or now >= end:
                event.state = 'close'
            elif now >= start and now < start_collab:
                event.state = 'competitive'
            elif now >= start and now >= start_collab and now < end:
                event.state = 'collab'
            if user:
                signed = get_event_team_by_name(
                                    db.session, event.name,
                                    flask_login.current_user.name)
                if not signed:
                    event.state_user = 'not_signed'
                elif signed.approved:
                    event.state_user = 'signed'
                elif signed:
                    event.state_user = 'waiting'
            else:
                event.state_user = 'not_signed'

    # problems = Problem.query.order_by(Problem.id.desc())
    return render_template('problems.html',
                           problems=problems,
                           admin=admin)


@mod.route("/problems/<problem_name>")
def problem(problem_name):
    """Landing page for a single RAMP problem.

    Parameters
    ----------
    problem_name : str
        The name of a problem.
    """
    current_problem = get_problem(db.session, problem_name)
    user = (flask_login.current_user
            if flask_login.current_user.is_authenticated else None)
    admin = user.access_level == 'admin' if user is not None else False
    if current_problem:
        if app.config['TRACK_USER_INTERACTION']:
            if flask_login.current_user.is_authenticated:
                add_user_interaction(
                    db.session,
                    interaction='looking at problem',
                    user=flask_login.current_user,
                    problem=current_problem
                )
            else:
                add_user_interaction(
                    db.session, interaction='looking at problem',
                    problem=current_problem
                )
        description_f_name = os.path.join(
            current_problem.path_ramp_kit,
            '{}_starting_kit.html'.format(current_problem.name)
        )
        # check which event ramp-kit archive is the latest
        archive_dir = os.path.join(
            current_problem.path_ramp_kit, "events_archived"
        )
        latest_event_zip = max(
            [f for f in os.scandir(archive_dir) if f.name.endswith(".zip")],
            key=lambda x: x.stat().st_mtime
        )
        latest_event = os.path.splitext(latest_event_zip.name)[0]

        return render_template(
            'problem.html', problem=current_problem, admin=admin,
            notebook_filename=description_f_name, latest_event=latest_event
        )
    else:
        return redirect_to_user('Problem {} does not exist'
                                .format(problem_name), is_error=True)


@mod.route("/download_starting_kit/<event_name>")
def download_starting_kit(event_name):
    event = db.session.query(Event).filter_by(name=event_name).one()
    return send_from_directory(
        os.path.join(event.problem.path_ramp_kit, "events_archived"),
        event_name + ".zip"
    )


@mod.route("/notebook/<problem_name>")
def notebook(problem_name):
    current_problem = get_problem(db.session, problem_name)
    return send_from_directory(
        current_problem.path_ramp_kit,
        '{}_starting_kit.html'.format(current_problem.name)
    )


@mod.route("/rules/<event_name>")
def rules(event_name):
    event = get_event(db.session, event_name)
    return render_template('rules.html', event=event)


@mod.route("/events/<event_name>")
@flask_login.login_required
def user_event(event_name):
    """Landing page for a given event.

    Parameters
    ----------
    event_name : str
        The event name.
    """
    if flask_login.current_user.access_level == 'asked':
        msg = 'Your account has not been approved yet by the administrator'
        logger.error(msg)
        return redirect_to_user(msg)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user('{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    event = get_event(db.session, event_name)
    if event:
        if app.config['TRACK_USER_INTERACTION']:
            add_user_interaction(db.session, interaction='looking at event',
                                 event=event, user=flask_login.current_user)
        admin = is_admin(db.session, event_name, flask_login.current_user.name)
        approved = is_user_signed_up(
            db.session, event_name, flask_login.current_user.name
        )
        asked = is_user_sign_up_requested(
            db.session, event_name, flask_login.current_user.name
        )
        return render_template('event.html',
                               event=event,
                               admin=admin,
                               approved=approved,
                               asked=asked)
    return redirect_to_user('Event {} does not exist.'
                            .format(event_name), is_error=True)


@mod.route("/events/<event_name>/sign_up")
@flask_login.login_required
def sign_up_for_event(event_name):
    """Landing page to sign-up to a specific RAMP event.

    Parameters
    ----------
    event_name : str
        The name of the event.
    """
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user('{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(db.session, interaction='signing up at event',
                             user=flask_login.current_user, event=event)

    ask_sign_up_team(db.session, event.name, flask_login.current_user.name)
    if event.is_controled_signup:
        admin_users = User.query.filter_by(access_level='admin')
        for admin in admin_users:
            subject = ('Request to sign-up {} to RAMP event {}'
                       .format(event.name, flask_login.current_user.name))
            body = body_formatter_user(flask_login.current_user)
            url_approve = ('http://{}/events/{}/sign_up/{}'
                           .format(
                               app.config['DOMAIN_NAME'], event.name,
                               flask_login.current_user.name
                           ))
            body += ('Click on this link to approve the sign-up request: {}'
                     .format(url_approve))
            send_mail(admin.email, subject, body)
        return redirect_to_user("Sign-up request is sent to event admins.",
                                is_error=False, category='Request sent')
    sign_up_team(db.session, event.name, flask_login.current_user.name)
    return redirect_to_sandbox(
        event,
        '{} is signed up for {}.'
        .format(flask_login.current_user.firstname, event),
        is_error=False,
        category='Successful sign-up'
    )


@mod.route("/events/<event_name>/sandbox", methods=['GET', 'POST'])
@flask_login.login_required
def sandbox(event_name):
    """Landing page for the user's sandbox.

    Parameters
    ----------
    event_name : str
        The event name.
    """
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            '{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if not is_accessible_code(db.session, event_name,
                              flask_login.current_user.name):
        error_str = ('No access to sandbox for event {}. If you have '
                     'already signed up, please wait for approval.'
                     .format(event.name))
        return redirect_to_user(error_str)
    # setup the webpage when loading
    # we use the code store in the sandbox to show to the user
    sandbox_submission = get_submission_by_name(
        db.session, event_name, flask_login.current_user.name,
        event.ramp_sandbox_name
    )
    event_team = get_event_team_by_name(
        db.session, event_name, flask_login.current_user.name
    )
    # initialize the form for the code
    # The amount of python magic we have to do for rendering a variable
    # number of textareas, named and populated at run time, is mind
    # boggling.

    # First we need to make sure CodeForm is empty
    # for name_code in CodeForm.names_codes:
    #     name, _ = name_code
    #     delattr(CodeForm, name)
    CodeForm.names_codes = []

    # Then we create named fields in the CodeForm class for each editable
    # submission file. They have to be populated when the code_form object
    # is created, so we also create a code_form_kwargs dictionary and
    # populate it with the codes.
    code_form_kwargs = {}
    for submission_file in sandbox_submission.files:
        if submission_file.is_editable:
            f_field = submission_file.name
            setattr(CodeForm,
                    f_field, StringField('Text', widget=TextArea()))
            code_form_kwargs[f_field] = submission_file.get_code()
    code_form_kwargs['prefix'] = 'code'
    code_form = CodeForm(**code_form_kwargs)
    # Then, to be able to iterate over the files in the sandbox.html
    # template, we also fill a separate table of pairs (file name, code).
    # The text areas in the template will then have to be created manually.
    for submission_file in sandbox_submission.files:
        if submission_file.is_editable:
            code_form.names_codes.append(
                (submission_file.name, submission_file.get_code()))

    # initialize the submission field and the the uploading form
    submit_form = SubmitForm(
        submission_name=event_team.last_submission_name, prefix='submit'
    )
    upload_form = UploadForm(prefix='upload')

    #  check if the event is before, during or after open state
    now = datetime.datetime.now()
    start = event.opening_timestamp
    end = event.closing_timestamp

    event_status = {"msg": "",
                    "state": "not_yet"}
    start_str = start.strftime("%d of %B %Y at %H:%M")
    end_str = end.strftime("%d of %B %Y, %H:%M")
    if now < start:
        event_status["msg"] = "Event submissions will open on the " + start_str
        event_status["state"] = "close"
    elif now < end:
        event_status["msg"] = "Event submissions are open until " + end_str
        event_status["state"] = "open"
    else:  # now >= end
        event_status["msg"] = "This event closed on the " + end_str
        event_status["state"] = "close"

    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    if request.method == 'GET':
        return render_template(
            'sandbox.html',
            submission_names=sandbox_submission.f_names,
            code_form=code_form,
            submit_form=submit_form, upload_form=upload_form,
            event=event,
            admin=admin,
            event_status=event_status
        )

    if request.method == 'POST':
        if ('code-csrf_token' in request.form and
                code_form.validate_on_submit()):
            try:
                for submission_file in sandbox_submission.files:
                    if submission_file.is_editable:
                        old_code = submission_file.get_code()
                        submission_file.set_code(
                            request.form[submission_file.name])
                        new_code = submission_file.get_code()
                        diff = '\n'.join(difflib.unified_diff(
                            old_code.splitlines(), new_code.splitlines()))
                        similarity = difflib.SequenceMatcher(
                            a=old_code, b=new_code).ratio()
                        if app.config['TRACK_USER_INTERACTION']:
                            add_user_interaction(
                                db.session,
                                interaction='save',
                                user=flask_login.current_user,
                                event=event,
                                submission_file=submission_file,
                                diff=diff, similarity=similarity
                            )
            except Exception as e:
                return redirect_to_sandbox(event, 'Error: {}'.format(e))

            # if we required to only save the file, redirect now
            if "saving" in request.form:
                return redirect_to_sandbox(
                    event,
                    'Your submission has been saved. You can safely comeback '
                    'to your sandbox later.',
                    is_error=False, category='File saved'
                )

        elif request.files:
            upload_f_name = secure_filename(
                request.files['file'].filename)
            upload_name = upload_f_name.split('.')[0]
            # TODO: create a get_function
            upload_workflow_element = WorkflowElement.query.filter_by(
                name=upload_name, workflow=event.workflow).one_or_none()
            if upload_workflow_element is None:
                return redirect_to_sandbox(event,
                                           '{} is not in the file list.'
                                           .format(upload_f_name))

            # TODO: create a get_function
            submission_file = SubmissionFile.query.filter_by(
                submission=sandbox_submission,
                workflow_element=upload_workflow_element).one()
            if submission_file.is_editable:
                old_code = submission_file.get_code()

            tmp_f_name = os.path.join(tempfile.gettempdir(), upload_f_name)
            request.files['file'].save(tmp_f_name)
            file_length = os.stat(tmp_f_name).st_size
            if (upload_workflow_element.max_size is not None and
                    file_length > upload_workflow_element.max_size):
                return redirect_to_sandbox(
                    event,
                    'File is too big: {} exceeds max size {}'
                    .format(file_length, upload_workflow_element.max_size)
                )
            if submission_file.is_editable:
                try:
                    with open(tmp_f_name) as f:
                        code = f.read()
                        submission_file.set_code(code)
                except Exception as e:
                    return redirect_to_sandbox(event, 'Error: {}'.format(e))
            else:
                # non-editable files are not verified for now
                dst = os.path.join(sandbox_submission.path, upload_f_name)
                shutil.copy2(tmp_f_name, dst)
            logger.info('{} uploaded {} in {}'
                        .format(flask_login.current_user.name, upload_f_name,
                                event))

            if submission_file.is_editable:
                new_code = submission_file.get_code()
                diff = '\n'.join(difflib.unified_diff(
                    old_code.splitlines(), new_code.splitlines()))
                similarity = difflib.SequenceMatcher(
                    a=old_code, b=new_code).ratio()
                if app.config['TRACK_USER_INTERACTION']:
                    add_user_interaction(
                        db.session,
                        interaction='upload',
                        user=flask_login.current_user,
                        event=event,
                        submission_file=submission_file,
                        diff=diff,
                        similarity=similarity
                    )
            else:
                if app.config['TRACK_USER_INTERACTION']:
                    add_user_interaction(
                        db.session,
                        interaction='upload',
                        user=flask_login.current_user,
                        event=event,
                        submission_file=submission_file
                    )

            return redirect(request.referrer)
            # TODO: handle different extensions for the same workflow element
            # ie: now we let upload eg external_data.bla, and only fail at
            # submission, without giving a message

        if 'submission' in request.form:
            if not submit_form.validate_on_submit():
                return redirect_to_sandbox(
                    event,
                    'Submission name should not contain any spaces'
                )
            new_submission_name = request.form['submit-submission_name']
            if not 4 < len(new_submission_name) < 20:
                return redirect_to_sandbox(
                    event,
                    'Submission name should have length between 4 and '
                    '20 characters.'
                )
            try:
                new_submission_name.encode('ascii')
            except Exception as e:
                return redirect_to_sandbox(event, 'Error: {}'.format(e))
            try:
                new_submission = add_submission(db.session, event_name,
                                                event_team.team.name,
                                                new_submission_name,
                                                sandbox_submission.path)
            except DuplicateSubmissionError:
                return redirect_to_sandbox(
                    event,
                    'Submission {} already exists. Please change the name.'
                    .format(new_submission_name)
                )

            except MissingExtensionError:
                return redirect_to_sandbox(
                    event, 'Missing extension'
                )
            except TooEarlySubmissionError as e:
                return redirect_to_sandbox(event, str(e))

            logger.info('{} submitted {} for {}.'
                        .format(flask_login.current_user.name,
                                new_submission.name, event_team))
            if event.is_send_submitted_mails:
                admin_users = User.query.filter_by(access_level='admin')
                for admin in admin_users:
                    subject = 'Submission {} sent for training'.format(
                        new_submission.name
                    )
                    body = """A new submission have been submitted:
                    event: {}
                    user: {}
                    submission: {}
                    submission path: {}
                    """.format(event_team.event.name,
                               flask_login.current_user.name,
                               new_submission.name, new_submission.path)
                    send_mail(admin.email, subject, body)
            if app.config['TRACK_USER_INTERACTION']:
                add_user_interaction(
                    db.session,
                    interaction='submit',
                    user=flask_login.current_user,
                    event=event,
                    submission=new_submission
                )

            return redirect_to_sandbox(
                event,
                '{} submitted {} for {}'
                .format(flask_login.current_user.firstname,
                        new_submission.name, event_team),
                is_error=False, category='Submission'
            )

    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    return render_template(
        'sandbox.html',
        submission_names=sandbox_submission.f_names,
        code_form=code_form,
        submit_form=submit_form, upload_form=upload_form,
        event=event,
        admin=admin,
        event_status=event_status
    )


@mod.route("/problems/<problem_name>/ask_for_event", methods=['GET', 'POST'])
@flask_login.login_required
def ask_for_event(problem_name):
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    if problem is None:
        return redirect_to_user(
            '{}: no problem named "{}"'
            .format(flask_login.current_user.firstname, problem_name)
        )
    logger.info('{} is asking for event on {}'
                .format(flask_login.current_user.name, problem.name))
    # We assume here that event name has the syntax <problem_name>_<suffix>
    form = AskForEventForm(
        min_duration_between_submissions_hour=8,
        min_duration_between_submissions_minute=0,
        min_duration_between_submissions_second=0,
    )
    if form.validate_on_submit():
        admin_users = User.query.filter_by(access_level='admin')
        for admin in admin_users:
            subject = 'Request to add a new event'
            body = """User {} asked to add a new event:
            event name: {}
            event title: {}
            number of students: {}
            waiting time between resubmission: {}:{}:{}
            opening data: {}
            closing data: {}
            """.format(
                flask_login.current_user.name,
                problem.name + '_' + form.suffix.data,
                form.title.data,
                form.n_students.data,
                form.min_duration_between_submissions_hour.data,
                form.min_duration_between_submissions_minute.data,
                form.min_duration_between_submissions_second.data,
                form.opening_date.data,
                form.closing_date.data
            )
            send_mail(admin.email, subject, body)
        return redirect_to_user(
            'Thank you. Your request has been sent to RAMP administrators.',
            category='Event request', is_error=False
        )

    return render_template('ask_for_event.html', form=form, problem=problem)


@mod.route("/credit/<submission_hash>", methods=['GET', 'POST'])
@flask_login.login_required
def credit(submission_hash):
    """The landing page to credit other submission when a user submit is own.

    Parameters
    ----------
    submission_hash : str
        The submission hash of the current submission.
    """
    submission = (Submission.query.filter_by(hash_=submission_hash)
                                  .one_or_none())
    access_code = is_accessible_code(
        db.session, submission.event_team.event.name,
        flask_login.current_user.name, submission.id
    )
    if submission is None or not access_code:
        error_str = 'Missing submission: {}'.format(submission_hash)
        return redirect_to_user(error_str)
    event_team = submission.event_team
    event = event_team.event
    source_submissions = get_source_submissions(db.session, submission.id)

    def get_s_field(source_submission):
        return '{}/{}/{}'.format(
            source_submission.event_team.event.name,
            source_submission.event_team.team.name,
            source_submission.name)

    # Make sure that CreditForm is empty
    CreditForm.name_credits = []
    credit_form_kwargs = {}
    for source_submission in source_submissions:
        s_field = get_s_field(source_submission)
        setattr(CreditForm, s_field, StringField('Text'))
    credit_form = CreditForm(**credit_form_kwargs)
    sum_credit = 0
    # new = True
    for source_submission in source_submissions:
        s_field = get_s_field(source_submission)
        submission_similaritys = \
            (SubmissionSimilarity.query
                                 .filter_by(
                                     type='target_credit',
                                     user=flask_login.current_user,
                                     source_submission=source_submission,
                                     target_submission=submission)
                                 .all())
        if not submission_similaritys:
            submission_credit = 0
        else:
            # new = False
            # find the last credit (in case crediter changes her mind)
            submission_similaritys.sort(
                key=lambda x: x.timestamp, reverse=True)
            submission_credit = int(
                round(100 * submission_similaritys[0].similarity)
            )
            sum_credit += submission_credit
        credit_form.name_credits.append(
            (s_field, str(submission_credit), source_submission.link)
        )
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
                return redirect_to_credit(
                    submission_hash,
                    'Error: The total credit should add up to 100'
                )
        except Exception as e:
            return redirect_to_credit(submission_hash, 'Error: {}'.format(e))
        for source_submission in source_submissions:
            s_field = get_s_field(source_submission)
            similarity = int(getattr(credit_form, s_field).data) / 100.
            submission_similarity = \
                (SubmissionSimilarity.query
                                     .filter_by(
                                         type='target_credit',
                                         user=flask_login.current_user,
                                         source_submission=source_submission,
                                         target_submission=submission)
                                     .all())
            # if submission_similarity is not empty, we need to
            # add zero to cancel previous credits explicitly
            if similarity > 0 or submission_similarity:
                add_submission_similarity(
                    db.session,
                    credit_type='target_credit',
                    user=flask_login.current_user,
                    source_submission=source_submission,
                    target_submission=submission,
                    similarity=similarity,
                    timestamp=datetime.datetime.utcnow()
                )

        if app.config['TRACK_USER_INTERACTION']:
            add_user_interaction(
                db.session,
                interaction='giving credit',
                user=flask_login.current_user,
                event=event,
                submission=submission
            )

        return redirect('/events/{}/sandbox'.format(event.name))

    admin = is_admin(db.session, event.name, flask_login.current_user.name)
    return render_template(
        'credit.html', submission=submission,
        source_submissions=source_submissions, credit_form=credit_form,
        event=event, admin=admin)


@mod.route("/event_plots/<event_name>")
@flask_login.login_required
def event_plots(event_name):
    """Landing page of the plot illustrating the score evolution over time for
    a specific RAMP event.

    Parameters
    ----------
    event_name : str
        The name of the event.
    """
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user('{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    if event:
        p = score_plot(db.session, event)
        script, div = components(p)
        return render_template('event_plots.html',
                               script=script,
                               div=div,
                               event=event)
    return redirect_to_user('Event {} does not exist.'
                            .format(event_name),
                            is_error=True)


@mod.route("/<submission_hash>/<f_name>", methods=['GET', 'POST'])
@flask_login.login_required
def view_model(submission_hash, f_name):
    """Rendering submission codes using templates/submission.html.

    The code of f_name is displayed in the left panel, the list of submissions
    files is in the right panel. Clicking on a file will show that file (using
    the same template). Clicking on the name on the top will download the file
    itself (managed in the template). Clicking on "Archive" will zip all the
    submission files and download them (managed here).

    Parameters
    ----------
    submission_hash : str
        The hash_ of the submission.
    f_name : tr
        The name of the submission file.
    """
    submission = (Submission.query.filter_by(hash_=submission_hash)
                                  .one_or_none())
    if (submission is None or
            not is_accessible_code(db.session, submission.event.name,
                                   flask_login.current_user.name,
                                   submission.id)):
        error_str = 'Missing submission: {}'.format(submission_hash)
        return redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    workflow_element_name = f_name.split('.')[0]
    workflow_element = \
        (WorkflowElement.query.filter_by(name=workflow_element_name,
                                         workflow=event.workflow)
                              .one_or_none())
    if workflow_element is None:
        error_str = ('{} is not a valid workflow element by {} '
                     .format(workflow_element_name,
                             flask_login.current_user.name))
        error_str += 'in {}/{}/{}/{}'.format(event, team, submission, f_name)
        return redirect_to_user(error_str)
    submission_file = \
        (SubmissionFile.query.filter_by(submission=submission,
                                        workflow_element=workflow_element)
                             .one_or_none())
    if submission_file is None:
        error_str = ('No submission file by {} in {}/{}/{}/{}'
                     .format(flask_login.current_user.name,
                             event, team, submission, f_name))
        return redirect_to_user(error_str)

    # superfluous, perhaps when we'll have different extensions?
    f_name = submission_file.f_name

    submission_abspath = os.path.abspath(submission.path)
    if not os.path.exists(submission_abspath):
        error_str = ('{} does not exist by {} in {}/{}/{}/{}'
                     .format(submission_abspath, flask_login.current_user.name,
                             event, team, submission, f_name))
        return redirect_to_user(error_str)

    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session,
            interaction='looking at submission',
            user=flask_login.current_user,
            event=event,
            submission=submission,
            submission_file=submission_file
        )

    logger.info('{} is looking at {}/{}/{}/{}'
                .format(flask_login.current_user.name, event, team, submission,
                        f_name))

    # Downloading file if it is not editable (e.g., external_data.csv)
    if not workflow_element.is_editable:
        # archive_filename = f_name  + '.zip'
        # with changedir(submission_abspath):
        #    with ZipFile(archive_filename, 'w') as archive:
        #        archive.write(f_name)
        if app.config['TRACK_USER_INTERACTION']:
            add_user_interaction(
                db.session,
                interaction='download',
                user=flask_login.current_user,
                event=event,
                submission=submission,
                submission_file=submission_file
            )

        return send_from_directory(
            submission_abspath, f_name, as_attachment=True,
            attachment_filename='{}_{}'.format(submission.hash_[:6], f_name),
            mimetype='application/octet-stream'
        )

    # Importing selected files into sandbox
    choices = [(f, f) for f in submission.f_names]
    import_form = ImportForm()
    import_form.selected_f_names.choices = choices
    if import_form.validate_on_submit():
        sandbox_submission = get_submission_by_name(
            db.session, event.name, flask_login.current_user.name,
            event.ramp_sandbox_name
        )
        for filename in import_form.selected_f_names.data:
            logger.info(
                '{} is importing {}/{}/{}/{}'
                .format(flask_login.current_user.name, event, team,
                        submission, filename)
            )

            workflow_element = WorkflowElement.query.filter_by(
                name=filename.split('.')[0], workflow=event.workflow).one()

            # TODO: deal with different extensions of the same file
            src = os.path.join(submission.path, filename)
            dst = os.path.join(sandbox_submission.path, filename)
            shutil.copy2(src, dst)  # copying also metadata
            logger.info('Copying {} to {}'.format(src, dst))

            submission_file = SubmissionFile.query.filter_by(
                submission=submission,
                workflow_element=workflow_element).one()
            if app.config['TRACK_USER_INTERACTION']:
                add_user_interaction(
                    db.session,
                    interaction='copy',
                    user=flask_login.current_user,
                    event=event,
                    submission=submission,
                    submission_file=submission_file
                )

        return redirect('/events/{}/sandbox'.format(event.name))

    with open(os.path.join(submission.path, f_name)) as f:
        code = f.read()
    admin = is_admin(db.session, event.name, flask_login.current_user.name)
    return render_template(
        'submission.html',
        event=event,
        code=code,
        submission=submission,
        f_name=f_name,
        import_form=import_form,
        admin=admin)


@mod.route("/<submission_hash>/error.txt")
@flask_login.login_required
def view_submission_error(submission_hash):
    """Rendering submission codes using templates/submission.html.

    The code of f_name is displayed in the left panel, the list of submissions
    files is in the right panel. Clicking on a file will show that file (using
    the same template). Clicking on the name on the top will download the file
    itself (managed in the template). Clicking on "Archive" will zip all the
    submission files and download them (managed here).

    Parameters
    ----------
    submission_hash : str
        The hash of the submission.
    """
    submission = (Submission.query.filter_by(hash_=submission_hash)
                                  .one_or_none())
    if submission is None:
        error_str = ('Missing submission {}: {}'
                     .format(flask_login.current_user.name, submission_hash))
        return redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    # TODO: check if event == submission.event_team.event

    if app.config['TRACK_USER_INTERACTION']:
        add_user_interaction(
            db.session,
            interaction='looking at error',
            user=flask_login.current_user,
            event=event,
            submission=submission
        )

    return render_template(
        'submission_error.html', submission=submission, team=team, event=event
    )


@mod.route("/toggle_competition/<submission_hash>")
@flask_login.login_required
def toggle_competition(submission_hash):
    """Pulling out or putting a submission back into competition.

    Parameters
    ----------
    submission_hash : str
        The submission hash of the current submission.
    """
    submission = (Submission.query.filter_by(hash_=submission_hash)
                                  .one_or_none())
    if submission is None:
        error_str = 'Missing submission: {}'.format(submission_hash)
        return redirect_to_user(error_str)

    access_code = is_accessible_code(
        db.session, submission.event_team.event.name,
        flask_login.current_user.name, submission.id
    )
    if not access_code:
        error_str = 'Missing submission: {}'.format(submission_hash)
        return redirect_to_user(error_str)

    submission.is_in_competition = not submission.is_in_competition
    db.session.commit()
    update_leaderboards(db.session, submission.event_team.event.name)
    return redirect(
        '/{}/{}'.format(submission_hash, submission.files[0].f_name)
    )


@mod.route("/download/<submission_hash>")
@flask_login.login_required
def download_submission(submission_hash):
    """Download a submission from the server.

    Parameters
    ----------
    submission_hash : str
        The submission hash of the current submission.
    """
    submission = (Submission.query.filter_by(hash_=submission_hash)
                                  .one_or_none())
    if submission is None:
        error_str = 'Missing submission: {}'.format(submission_hash)
        return redirect_to_user(error_str)

    access_code = is_accessible_code(
        db.session, submission.event_team.event.name,
        flask_login.current_user.name, submission.id
    )
    if not access_code:
        error_str = 'Unauthorized access: {}'.format(submission_hash)
        return redirect_to_user(error_str)

    file_in_memory = io.BytesIO()
    with zipfile.ZipFile(file_in_memory, 'w') as zf:
        for ff in submission.files:
            data = zipfile.ZipInfo(ff.f_name)
            data.date_time = time.localtime(time.time())[:6]
            data.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(data, ff.get_code())
    file_in_memory.seek(0)
    return send_file(
        file_in_memory,
        attachment_filename=f"submission_{submission.id}.zip",
        as_attachment=True,
    )
