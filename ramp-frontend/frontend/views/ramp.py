import codecs
import difflib
import logging
import os
import shutil
import tempfile

import flask_login

from flask import Blueprint
from flask import flash
from flask import render_template
from flask import request
from flask import redirect

from wtforms import StringField
from wtforms.widgets import TextArea

from werkzeug.utils import secure_filename

from rampdb.model import EventTeam
from rampdb.model import SubmissionFile
from rampdb.model import User
from rampdb.model import WorkflowElement

from rampdb.exceptions import DuplicateSubmissionError
from rampdb.exceptions import MissingExtensionError
from rampdb.exceptions import TooEarlySubmissionError

from rampdb.tools.event import get_event
from rampdb.tools.event import get_problem
from rampdb.tools.event import is_accessible_event
from rampdb.tools.event import is_admin
from rampdb.tools.submission import add_submission
from rampdb.tools.submission import is_accessible_code
from rampdb.tools.submission import get_submission_by_name
from rampdb.tools.user import add_user_interaction
from rampdb.tools.user import approve_user
from rampdb.tools.user import get_user_by_name
from rampdb.tools.team import ask_sign_up_team
from rampdb.tools.team import get_event_team_by_name
from rampdb.tools.team import sign_up_team
from rampdb.tools.team import is_user_signed_up

from frontend import db

from ..forms import CodeForm
from ..forms import SubmitForm
from ..forms import UploadForm

from .redirect import redirect_to_sandbox
from .redirect import redirect_to_user

mod = Blueprint('ramp', __name__)
logger = logging.getLogger('FRONTEND')


@mod.route("/problems")
def problems():
    """Problems request."""
    user = (flask_login.current_user
            if flask_login.current_user.is_authenticated else None)
    add_user_interaction(
        db.session, interaction='looking at problems', user=user
    )

    # problems = Problem.query.order_by(Problem.id.desc())
    return render_template('problems.html',
                           problems=get_problem(db.session, None))


@mod.route("/problems/<problem_name>")
def problem(problem_name):
    problem = get_problem(db.session, problem_name)
    if problem:
        if flask_login.current_user.is_authenticated:
            add_user_interaction(
                db.session,
                interaction='looking at problem',
                user=flask_login.current_user,
                problem=problem
            )
        else:
            add_user_interaction(
                db.session, interaction='looking at problem', problem=problem
            )
        description_f_name = os.path.join(
            problem.path_ramp_kits, problem.name,
            '{}_starting_kit.html'.format(problem.name)
        )
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        return render_template('problem.html', problem=problem,
                               description=description)
    else:
        return redirect_to_user(u'Problem {} does not exist.'
                                .format(problem_name), is_error=True)


@mod.route("/events/<event_name>")
@flask_login.login_required
def user_event(event_name):
    if flask_login.current_user.access_level == 'asked':
        msg = 'Your account has not been approved yet by the administrator'
        logger.error(msg)
        return redirect_to_user(msg)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(u'{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    event = get_event(db.session, event_name)
    if event:
        if flask_login.current_user.is_authenticated:
            add_user_interaction(db.session, interaction='looking at event',
                                 user=flask_login.current_user, event=event)
        else:
            add_user_interaction(db.session, interaction='looking at event',
                                 event=event)
        description_f_name = os.path.join(
            event.problem.path_ramp_kits,
            event.problem.name,
            '{}_starting_kit.html'.format(event.problem.name)
        )
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        admin = is_admin(db.session, event_name, flask_login.current_user.name)
        if flask_login.current_user.is_anonymous:
            approved = False
            asked = False
        else:
            approved = is_user_signed_up(
                db.session, event_name, flask_login.current_user.name
            )
            asked = approved
        return render_template('event.html',
                               description=description,
                               event=event,
                               admin=admin,
                               approved=approved,
                               asked=asked)
    return redirect_to_user(u'Event {} does not exist.'
                            .format(event_name), is_error=True)


@mod.route("/events/<event_name>/sign_up")
@flask_login.login_required
def sign_up_for_event(event_name):
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(u'{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    add_user_interaction(db.session, interaction='signing up at event',
                         user=flask_login.current_user, event=event)

    ask_sign_up_team(db.session, event.name, flask_login.current_user.name)
    if event.is_controled_signup:
        # send_sign_up_request_mail(event, flask_login.current_user)
        return redirect_to_user("Sign-up request is sent to event admins.",
                                is_error=False, category='Request sent')
    sign_up_team(db.session, event.name, flask_login.current_user.name)
    return redirect_to_sandbox(
        event,
        u'{} is signed up for {}.'
        .format(flask_login.current_user.firstname, event),
        is_error=False,
        category='Successful sign-up'
    )


@mod.route("/events/<event_name>/my_submissions")
@flask_login.login_required
def my_submissions(event_name):
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            u'{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    add_user_interaction(
        db.session, interaction='looking at my_submissions',
        user=flask_login.current_user, event=event
    )
    if not is_accessible_code(db.session, event_name,
                              flask_login.current_user.name):
        error_str = ('No access to my submissions for event {}. If you have '
                     'already signed up, please wait for approval.'
                     .format(event.name))
        return redirect_to_user(error_str)

    # Doesn't work if team mergers are allowed
    event_team = get_event_team_by_name(db.session, event_name,
                                        flask_login.current_user.name)
    leaderboard_html = event_team.leaderboard_html
    failed_leaderboard_html = event_team.failed_leaderboard_html
    new_leaderboard_html = event_team.new_leaderboard_html
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
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


@mod.route("/events/<event_name>/sandbox", methods=['GET', 'POST'])
@flask_login.login_required
def sandbox(event_name):
    if request.method == 'POST':
        print(request.form)
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            u'{}: no access or no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if not is_accessible_code(db.session, event_name,
                              flask_login.current_user.name):
        error_str = ('No access to sandbox for event {}. If you have '
                     'already signed up, please wait for approval.'
                     .format(event.name))
        return redirect_to_user(error_str)

    sandbox_submission = get_submission_by_name(db.session, event_name,
                                                flask_login.current_user.name,
                                                event.ramp_sandbox_name)
    event_team = get_event_team_by_name(db.session, event_name,
                                        flask_login.current_user.name)

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

    # print(code_form.names_codes)
    # print(getattr(code_form, code_form.names_codes[0][0]).data)
    if (code_form.names_codes and
            code_form.validate_on_submit() and
            getattr(code_form, code_form.names_codes[0][0]).data):
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
                    add_user_interaction(
                        db.session,
                        interaction='save',
                        user=flask_login.current_user,
                        event=event,
                        submission_file=submission_file,
                        diff=diff, similarity=similarity
                    )
        except Exception as e:
            return redirect_to_sandbox(event, u'Error: {}'.format(e))
        return redirect_to_sandbox(
            event,
            u'{} saved submission files for {}.'
            .format(flask_login.current_user.name, event),
            is_error=False,
            category='File saved'
        )

    if upload_form.validate_on_submit() and upload_form.file.data:
        upload_f_name = secure_filename(upload_form.file.data.filename)
        upload_name = upload_f_name.split('.')[0]
        # TODO: create a get_function
        upload_workflow_element = WorkflowElement.query.filter_by(
            name=upload_name, workflow=event.workflow).one_or_none()
        if upload_workflow_element is None:
            return redirect_to_sandbox(event,
                                       u'{} is not in the file list.'
                                       .format(upload_f_name))

        # TODO: create a get_function
        submission_file = SubmissionFile.query.filter_by(
            submission=sandbox_submission,
            workflow_element=upload_workflow_element).one()
        if submission_file.is_editable:
            old_code = submission_file.get_code()

        tmp_f_name = os.path.join(tempfile.gettempdir(), upload_f_name)
        upload_form.file.data.save(tmp_f_name)
        file_length = os.stat(tmp_f_name).st_size
        if (upload_workflow_element.max_size is not None and
                file_length > upload_workflow_element.max_size):
            return redirect_to_sandbox(
                event,
                u'File is too big: {} exceeds max size {}'
                .format(file_length, upload_workflow_element.max_size)
            )
        if submission_file.is_editable:
            try:
                with open(tmp_f_name) as f:
                    code = f.read()
                    submission_file.set_code(code)  # to verify eg asciiness
            except Exception as e:
                return redirect_to_sandbox(event, u'Error: {}'.format(e))
        else:
            # non-editable files are not verified for now
            dst = os.path.join(sandbox_submission.path, upload_f_name)
            shutil.copy2(tmp_f_name, dst)
        logger.info(u'{} uploaded {} in {}'
                    .format(flask_login.current_user.name, upload_f_name,
                            event))

        if submission_file.is_editable:
            new_code = submission_file.get_code()
            diff = '\n'.join(difflib.unified_diff(
                old_code.splitlines(), new_code.splitlines()))
            similarity = difflib.SequenceMatcher(
                a=old_code, b=new_code).ratio()
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

    # print(submit_form.is_submitted())
    # print(submit_form.validate())
    # print(submit_form.validate_on_submit())
    # print(submit_form.submission_name.data)
    if submit_form.validate_on_submit() and submit_form.submission_name.data:
        new_submission_name = submit_form.submission_name.data
        if not 4 > len(new_submission_name) > 20:
            return redirect_to_sandbox(event, 'Submission name should have '
                                       'length between 4 and 20 characters.')
        try:
            new_submission_name.encode('ascii')
        except Exception as e:
            return redirect_to_sandbox(event, u'Error: {}'.format(e))

        try:
            new_submission = add_submission(db.session, event_name,
                                            event_team.team.name,
                                            new_submission_name,
                                            sandbox_submission.path)
            if os.path.exists(new_submission.path):
                shutil.rmtree(new_submission.path)
            os.makedirs(new_submission.path)
            from_submission_path = os.path.join(sandbox_submission.path,
                                                new_submission_name)
            for filename in new_submission.f_names:
                shutil.copy2(src=os.path.join(from_submission_path, filename),
                             dst=os.path.join(new_submission.path, filename))
        except DuplicateSubmissionError:
            return redirect_to_sandbox(
                event,
                u'Submission {} already exists. Please change the name.'
                .format(new_submission_name)
            )
        except MissingExtensionError as e:
            return redirect_to_sandbox(
                event, 'Missing extension'
            )
        except TooEarlySubmissionError as e:
            return redirect_to_sandbox(event, str(e))

        logger.info(u'{} submitted {} for {}.'.format(
            flask_login.current_user.name, new_submission.name, event_team))
        # if event.is_send_submitted_mails:
        #     try:
        #         send_submission_mails(
        #             flask_login.current_user, new_submission, event_team)
        #     except Exception as e:
        #         error_str = u'mail was not sent {} '.format(
        #             flask_login.current_user.name)
        #         error_str += u'submitted {} for {}\n{}.'.format(
        #             new_submission.name, event_team, e)
        #         logger.error(error_str)
        flash(u'{} submitted {} for {}.'
              .format(flask_login.current_user.firstname, new_submission.name,
                      event_team),
              category='Submission')

        add_user_interaction(
            db.session,
            interaction='submit',
            user=flask_login.current_user,
            event=event,
            submission=new_submission
        )

        return redirect(u'/credit/{}'.format(new_submission.hash_))

    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    return render_template(
        'sandbox.html',
        submission_names=sandbox_submission.f_names,
        code_form=code_form,
        submit_form=submit_form, upload_form=upload_form,
        event=event,
        admin=admin
    )


# TODO: Function that should go in an admin blueprint and admin board
@mod.route("/approve_users", methods=['GET', 'POST'])
@flask_login.login_required
def approve_users():
    if not flask_login.current_user.access_level == 'admin':
        return redirect_to_user(
            u'Sorry {}, you do not have admin rights'
            .format(flask_login.current_user.firstname),
            is_error=True
        )
    if request.method == 'GET':
        # TODO: replace by some get_functions
        asked_users = User.query.filter_by(access_level='asked').all()
        asked_sign_up = EventTeam.query.filter_by(approved=False).all()
        return render_template('approve.html',
                               asked_users=asked_users,
                               asked_sign_up=asked_sign_up,
                               admin=True)
    users_to_be_approved = request.form.getlist('approve_users')
    event_teams_to_be_approved = request.form.getlist('approve_event_teams')
    message = "Approved users:\n"
    for asked_user in users_to_be_approved:
        approve_user(db.session, asked_user)
        message += "{}\n".format(asked_user)
    message += "Approved event_team:\n"
    for asked_id in event_teams_to_be_approved:
        asked_event_team = EventTeam.query.get(asked_id)
        sign_up_team(db.session, asked_event_team.event.name,
                     asked_event_team.team.name)
        message += "{}\n".format(asked_event_team)
    return redirect_to_user(message, is_error=False, category="Approved users")


@mod.route("/events/<event_name>/sign_up/<user_name>")
@flask_login.login_required
def approve_sign_up_for_event(event_name, user_name):
    event = get_event(db.session, event_name)
    user = get_user_by_name(db.session, user_name)
    if not is_admin(db.session, event_name, flask_login.current_user.name):
        return redirect_to_user(u'Sorry {}, you do not have admin rights'
                                .format(flask_login.current_user.firstname),
                                is_error=True)
    if not event or not user:
        return redirect_to_user(u'Oups, no event {} or no user {}.'
                                .format(event_name, user_name), is_error=True)
    sign_up_team(db.session, event.name, user.name)
    return redirect_to_user(u'{} is signed up for {}.'.format(user, event),
                            is_error=False, category='Successful sign-up')
