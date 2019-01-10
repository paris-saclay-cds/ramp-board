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
from flask import send_from_directory
from flask import url_for

from sqlalchemy.exc import IntegrityError

from wtforms import StringField
from wtforms.widgets import TextArea

from werkzeug.utils import secure_filename

from rampdb.model import EventTeam
from rampdb.model import Submission
from rampdb.model import SubmissionFile
from rampdb.model import User
from rampdb.model import WorkflowElement

from rampdb.exceptions import DuplicateSubmissionError
from rampdb.exceptions import MissingExtensionError
from rampdb.exceptions import NameClashError
from rampdb.exceptions import TooEarlySubmissionError

from rampdb.tools.event import get_event
from rampdb.tools.event import get_problem
from rampdb.tools.event import is_accessible_event
from rampdb.tools.event import is_accessible_leaderboard
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
from ..forms import EventUpdateProfileForm
from ..forms import ImportForm
from ..forms import SubmitForm
from ..forms import UploadForm

from .redirect import redirect_to_sandbox
from .redirect import redirect_to_user

from .visualization import score_plot

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
                    f_field, StringField(u'Text', widget=TextArea()))
            code_form_kwargs[f_field] = submission_file.get_code()
    code_form = CodeForm(**code_form_kwargs, prefix="code")
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

    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    if request.method == 'GET':
        return render_template(
            'sandbox.html',
            submission_names=sandbox_submission.f_names,
            code_form=code_form,
            submit_form=submit_form, upload_form=upload_form,
            event=event,
            admin=admin
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
            return render_template(
                'sandbox.html',
                submission_names=sandbox_submission.f_names,
                code_form=code_form,
                submit_form=submit_form,
                upload_form=upload_form,
                event=event,
                admin=admin
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
                                           u'{} is not in the file list.'
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
                    u'File is too big: {} exceeds max size {}'
                    .format(file_length, upload_workflow_element.max_size)
                )
            if submission_file.is_editable:
                try:
                    with open(tmp_f_name) as f:
                        code = f.read()
                        submission_file.set_code(code)
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

        elif ('submit-csrf_token' in request.form and
              submit_form.validate_on_submit()):
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
                return redirect_to_sandbox(event, u'Error: {}'.format(e))

            try:
                new_submission = add_submission(db.session, event_name,
                                                event_team.team.name,
                                                new_submission_name,
                                                sandbox_submission.path)
                if os.path.exists(new_submission.path):
                    shutil.rmtree(new_submission.path)
                os.makedirs(new_submission.path)
                for filename in new_submission.f_names:
                    shutil.copy2(
                        src=os.path.join(sandbox_submission.path, filename),
                        dst=os.path.join(new_submission.path, filename)
                    )
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

            logger.info(u'{} submitted {} for {}.'
                        .format(flask_login.current_user.name,
                                new_submission.name, event_team))
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
                  .format(flask_login.current_user.firstname,
                          new_submission.name,
                          event_team),
                  category='Submission')

            add_user_interaction(
                db.session,
                interaction='submit',
                user=flask_login.current_user,
                event=event,
                submission=new_submission
            )

            # return redirect(u'/credit/{}'.format(new_submission.hash_))
            return render_template(
                'sandbox.html',
                submission_names=sandbox_submission.f_names,
                code_form=code_form,
                submit_form=submit_form, upload_form=upload_form,
                event=event,
                admin=admin
            )

    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    return render_template(
        'sandbox.html',
        submission_names=sandbox_submission.f_names,
        code_form=code_form,
        submit_form=submit_form, upload_form=upload_form,
        event=event,
        admin=admin
    )


@mod.route("/events/<event_name>/leaderboard")
@flask_login.login_required
def leaderboard(event_name):
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            u'{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name))
    add_user_interaction(
        db.session,
        interaction='looking at leaderboard',
        user=flask_login.current_user,
        event=event
    )

    if is_accessible_leaderboard(db.session, event_name,
                                 flask_login.current_user.name):
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

    if is_admin(db.session, event_name, flask_login.current_user.name):
        failed_leaderboard_html = event.failed_leaderboard_html
        new_leaderboard_html = event.new_leaderboard_html
        template = render_template(
            'leaderboard.html',
            failed_leaderboard=failed_leaderboard_html,
            new_leaderboard=new_leaderboard_html,
            admin=True,
            **leaderboard_kwargs
        )
    else:
        template = render_template(
            'leaderboard.html', **leaderboard_kwargs
        )

    return template


@mod.route("/events/<event_name>/competition_leaderboard")
@flask_login.login_required
def competition_leaderboard(event_name):
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            u'{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    add_user_interaction(
        db.session,
        interaction='looking at leaderboard',
        user=flask_login.current_user,
        event=event
    )
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
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


@mod.route("/events/<event_name>/update", methods=['GET', 'POST'])
@flask_login.login_required
def update_event(event_name):
    if not flask_login.current_user.is_authenticated:
        return redirect(url_for('login'))
    event = get_event(db.session, event_name)
    if not is_accessible_event(db.session, event_name,
                               flask_login.current_user.name):
        return redirect_to_user(
            u'{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if not is_admin(db.session, event_name, flask_login.current_user.name):
        return redirect(url_for('problems'))
    logger.info(u'{} is updating event {}'
                .format(flask_login.current_user.name, event.name))
    admin = is_admin(db.session, event_name, flask_login.current_user.name)
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
            existing_event = get_event(db.session, event.name)
            if existing_event is not None:
                message += 'event name is already in use'
            # # try:
            # #     User.query.filter_by(email=email).one()
            # #     if len(message) > 0:
            # #         message += ' and '
            # #     message += 'email is already in use'
            # except NoResultFound:
            #     pass
            if message:
                e = NameClashError(message)
            flash(u'{}'.format(e), category='Update event error')
            return redirect(url_for('update_event', event_name=event.name))

        return redirect(url_for('ramp.problems'))

    return render_template(
        'update_event.html',
        form=form,
        event=event,
        admin=admin,
    )


@mod.route("/event_plots/<event_name>")
@flask_login.login_required
def event_plots(event_name):
    from bokeh.embed import components
    from bokeh.plotting import show
    event = get_event(db.session, event_name)
    if not is_accessible_code(db.session, event_name,
                              flask_login.current_user.name):
        return redirect_to_user(
            u'{}: no event named "{}"'
            .format(flask_login.current_user.firstname, event_name)
        )
    if event:
        p = score_plot(db.session, event)
        script, div = components(p)
        return render_template('event_plots.html',
                               script=script,
                               div=div,
                               event=event)
    return redirect_to_user(u'Event {} does not exist.'
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
    submission_hash : string
        The hash_ of the submission.
    f_name : string
        The name of the submission file

    Returns
    -------
     : html string
        The rendered submission.html page.
    """
    submission = (Submission.query.filter_by(hash_=submission_hash)
                                  .one_or_none())
    if (submission is None or
            not is_accessible_code(db.session, submission.event_name,
                                   flask_login.current_user.name,
                                   submission.name)):
        error_str = u'Missing submission: {}'.format(submission_hash)
        return redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    workflow_element_name = f_name.split('.')[0]
    workflow_element = \
        (WorkflowElement.query.filter_by(name=workflow_element_name,
                                         workflow=event.workflow)
                              .one_or_none())
    if workflow_element is None:
        error_str = (u'{} is not a valid workflow element by {} '
                     .format(workflow_element_name,
                             flask_login.current_user.name))
        error_str += u'in {}/{}/{}/{}'.format(event, team, submission, f_name)
        return redirect_to_user(error_str)
    submission_file = \
        (SubmissionFile.query.filter_by(submission=submission,
                                        workflow_element=workflow_element)
                             .one_or_none())
    if submission_file is None:
        error_str = (u'No submission file by {} in {}/{}/{}/{}'
                     .format(flask_login.current_user.name,
                             event, team, submission, f_name))
        return redirect_to_user(error_str)

    # superfluous, perhaps when we'll have different extensions?
    f_name = submission_file.f_name

    submission_abspath = os.path.abspath(submission.path)
    if not os.path.exists(submission_abspath):
        error_str = (u'{} does not exist by {} in {}/{}/{}/{}'
                     .format(submission_abspath, flask_login.current_user.name,
                             event, team, submission, f_name))
        return redirect_to_user(error_str)

    add_user_interaction(
        db.session,
        interaction='looking at submission',
        user=flask_login.current_user,
        event=event,
        submission=submission,
        submission_file=submission_file
    )

    logger.info(u'{} is looking at {}/{}/{}/{}'
                .format(flask_login.current_user.name, event, team, submission,
                        f_name))

    # Downloading file if it is not editable (e.g., external_data.csv)
    if not workflow_element.is_editable:
        # archive_filename = f_name  + '.zip'
        # with changedir(submission_abspath):
        #    with ZipFile(archive_filename, 'w') as archive:
        #        archive.write(f_name)
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
            attachment_filename=u'{}_{}'.format(submission.hash_[:6], f_name),
            mimetype='application/octet-stream'
        )

    # Importing selected files into sandbox
    choices = [(f, f) for f in submission.f_names]
    import_form = ImportForm()
    import_form.selected_f_names.choices = choices
    if import_form.validate_on_submit():
        sandbox_submission = get_sandbox(event, flask_login.current_user)
        for f_name in import_form.selected_f_names.data:
            logger.info(u'{} is importing {}/{}/{}/{}'.format(
                flask_login.current_user.name, event, team, submission, f_name))

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
            add_user_interaction(
                interaction='copy', user=flask_login.current_user, event=event,
                submission=submission, submission_file=submission_file)

        return redirect(u'/events/{}/sandbox'.format(event.name))

    with open(os.path.join(submission.path, f_name)) as f:
        code = f.read()
    admin = check_admin(flask_login.current_user, event)
    return render_template(
        'submission.html',
        event=event,
        code=code,
        submission=submission,
        f_name=f_name,
        import_form=import_form,
        admin=admin)


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
