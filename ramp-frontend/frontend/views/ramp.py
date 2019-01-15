import codecs
import datetime
import difflib
import logging
import os
import shutil
import tempfile

from bokeh.embed import components

import flask_login

from flask import Blueprint
from flask import flash
from flask import redirect
from flask import render_template
from flask import request
from flask import send_from_directory

from wtforms import StringField
from wtforms.widgets import TextArea

from werkzeug.utils import secure_filename

from rampdb.model import Submission
from rampdb.model import SubmissionFile
from rampdb.model import SubmissionSimilarity
from rampdb.model import WorkflowElement

from rampdb.exceptions import DuplicateSubmissionError
from rampdb.exceptions import MissingExtensionError
from rampdb.exceptions import TooEarlySubmissionError

from rampdb.tools.event import get_event
from rampdb.tools.event import get_problem
from rampdb.tools.frontend import is_admin
from rampdb.tools.frontend import is_accessible_code
from rampdb.tools.frontend import is_accessible_event
from rampdb.tools.frontend import is_user_signed_up
from rampdb.tools.submission import add_submission
from rampdb.tools.submission import add_submission_similarity
from rampdb.tools.submission import get_source_submissions
from rampdb.tools.submission import get_submission_by_name
from rampdb.tools.user import add_user_interaction
from rampdb.tools.team import ask_sign_up_team
from rampdb.tools.team import get_event_team_by_name
from rampdb.tools.team import sign_up_team

from frontend import db

from ..forms import CodeForm
from ..forms import CreditForm
from ..forms import ImportForm
from ..forms import SubmitForm
from ..forms import UploadForm

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
    add_user_interaction(
        db.session, interaction='looking at problems', user=user
    )

    # problems = Problem.query.order_by(Problem.id.desc())
    return render_template('problems.html',
                           problems=get_problem(db.session, None))


@mod.route("/problems/<problem_name>")
def problem(problem_name):
    """Landing page for a single RAMP problem.

    Parameters
    ----------
    problem_name : str
        The name of a problem.
    """
    current_problem = get_problem(db.session, problem_name)
    if current_problem:
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
            current_problem.path_ramp_kits, current_problem.name,
            '{}_starting_kit.html'.format(current_problem.name)
        )
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        return render_template('problem.html', problem=current_problem,
                               description=description)
    else:
        return redirect_to_user(u'Problem {} does not exist'
                                .format(problem_name), is_error=True)


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
        return redirect_to_user(u'{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
    event = get_event(db.session, event_name)
    if event:
        add_user_interaction(db.session, interaction='looking at event',
                             event=event, user=flask_login.current_user)
        description_f_name = os.path.join(
            event.problem.path_ramp_kits,
            event.problem.name,
            '{}_starting_kit.html'.format(event.problem.name)
        )
        with codecs.open(description_f_name, 'r', 'utf-8') as description_file:
            description = description_file.read()
        admin = is_admin(db.session, event_name, flask_login.current_user.name)
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
    """Landing page to sign-up to a specific RAMP event.

    Parameters
    ----------
    event_name : str
        The name of the event.
    """
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
            u'{}: no event named "{}"'
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

            return redirect(u'/credit/{}'.format(new_submission.hash_))
            # return render_template(
            #     'sandbox.html',
            #     submission_names=sandbox_submission.f_names,
            #     code_form=code_form,
            #     submit_form=submit_form, upload_form=upload_form,
            #     event=event,
            #     admin=admin
            # )

    admin = is_admin(db.session, event_name, flask_login.current_user.name)
    return render_template(
        'sandbox.html',
        submission_names=sandbox_submission.f_names,
        code_form=code_form,
        submit_form=submit_form, upload_form=upload_form,
        event=event,
        admin=admin
    )


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
        flask_login.current_user.name, submission.name
    )
    if submission is None or not access_code:
        error_str = u'Missing submission: {}'.format(submission_hash)
        return redirect_to_user(error_str)
    event_team = submission.event_team
    event = event_team.event
    source_submissions = get_source_submissions(db.session, submission.id)

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
            submission_credit = int(round(100 * submission_similaritys[0].similarity))
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
            return redirect_to_credit(submission_hash, u'Error: {}'.format(e))
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

        add_user_interaction(
            db.session,
            interaction='giving credit',
            user=flask_login.current_user,
            event=event,
            submission=submission
        )

        return redirect(u'/events/{}/sandbox'.format(event.name))

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
        return redirect_to_user(u'{}: no event named "{}"'
                                .format(flask_login.current_user.firstname,
                                        event_name))
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
        sandbox_submission = get_submission_by_name(
            db.session, event.name, flask_login.current_user.name,
            event.ramp_sandbox_name
        )
        for filename in import_form.selected_f_names.data:
            logger.info(
                u'{} is importing {}/{}/{}/{}'
                .format(flask_login.current_user.name, event, team,
                        submission, filename)
            )

            # TODO: deal with different extensions of the same file
            src = os.path.join(submission.path, filename)
            dst = os.path.join(sandbox_submission.path, filename)
            shutil.copy2(src, dst)  # copying also metadata
            logger.info(u'Copying {} to {}'.format(src, dst))

            workflow_element = WorkflowElement.query.filter_by(
                name=filename.split('.')[0], workflow=event.workflow).one()
            submission_file = SubmissionFile.query.filter_by(
                submission=submission,
                workflow_element=workflow_element).one()
            add_user_interaction(
                db.session,
                interaction='copy',
                user=flask_login.current_user,
                event=event,
                submission=submission,
                submission_file=submission_file
            )

        return redirect(u'/events/{}/sandbox'.format(event.name))

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
        error_str = (u'Missing submission {}: {}'
                     .format(flask_login.current_user.name, submission_hash))
        return redirect_to_user(error_str)
    event = submission.event_team.event
    team = submission.event_team.team
    # TODO: check if event == submission.event_team.event

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
