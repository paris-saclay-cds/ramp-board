import os
import shutil
import bcrypt
import timeit
import smtplib
import logging
import datetime
import numpy as np
import pandas as pd
import xkcdpass.xkcd_password as xp
from sklearn.externals.joblib import Parallel, delayed
from databoard import db

from databoard.model import User, Team, Submission, SubmissionFile,\
    SubmissionFileType, SubmissionFileTypeExtension, WorkflowElementType,\
    WorkflowElement, Workflow, Extension, Problem, Event, EventTeam,\
    CVFold, SubmissionOnCVFold, DetachedSubmissionOnCVFold,\
    UserInteraction,\
    NameClashError, MergeTeamError, TooEarlySubmissionError,\
    DuplicateSubmissionError, MissingSubmissionFileError,\
    MissingExtensionError,\
    combine_predictions_list, get_next_best_single_fold,\
    get_active_user_event_team, get_user_teams, get_n_team_members,\
    get_team_members
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

import databoard.config as config
import databoard.generic as generic


logger = logging.getLogger('databoard')
pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output


def date_time_format(date_time):
    return date_time.strftime('%Y-%m-%d %H:%M:%S %a')


######################### Users ###########################

def get_hashed_password(plain_text_password):
    """Hash a password for the first time
    (Using bcrypt, the salt is saved into the hash itself)
    """
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hased password. Using bcrypt, the salt is saved into the
    hash itself
    """
    return bcrypt.checkpw(plain_text_password, hashed_password)


def generate_single_password(mywords=None):
    if mywords is None:
        words = xp.locate_wordfile()
        mywords = xp.generate_wordlist(
            wordfile=words, min_length=4, max_length=6)
    return xp.generate_xkcdpassword(mywords, numwords=4)


def generate_passwords(users_to_add_f_name, password_f_name):
    users_to_add = pd.read_csv(users_to_add_f_name)
    words = xp.locate_wordfile()
    mywords = xp.generate_wordlist(wordfile=words, min_length=4, max_length=6)
    users_to_add['password'] = [
        generate_single_password(mywords) for name in users_to_add['name']]
    # temporarily while we don't implement pwd recovery
    users_to_add[['name', 'password']].to_csv(password_f_name, index=False)


def send_password_mail(user_name, password, port=None):
    """Also resets password. If port is None, use
    config.config_object.server_port.
    """

    user = User.query.filter_by(name=user_name).one()
    user.hashed_password = get_hashed_password(password)
    db.session.commit()

    gmail_user = config.MAIL_USERNAME
    gmail_pwd = config.MAIL_PASSWORD
    smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)

    logger.info('Sending mail to {}'.format(user.email))
    subject = '{} RAMP information'.format(
        config.config_object.specific.ramp_title)
    header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
        user.email, gmail_user, subject)
    body = 'Dear {},\n\n'.format(user.firstname)
    body += 'Here is your login and other information for the {} RAMP:\n\n'.format(
        config.config_object.specific.ramp_title)
    body += 'username: {}\n'.format(user.name)
    body += 'password: {}\n'.format(password)

    if port is None:
        body += 'submission site: http://{}:{}\n'.format(
            config.config_object.web_server, config.config_object.server_port)
    elif port == '80':
        body += 'submission site: http://{}\n'.format(
            config.config_object.web_server)
    else:
        body += 'submission site: http://{}:{}\n'.format(
            config.config_object.web_server, port)

    if config.opening_timestamp is not None:
        body += 'opening at (UTC) {}\n'.format(
            date_time_format(config.opening_timestamp))
    if config.public_opening_timestamp is not None:
        body += 'opening of the collaborative phase at (UTC) {}\n'.format(
            date_time_format(config.public_opening_timestamp))
    if config.closing_timestamp is not None:
        body += 'closing at (UTC) {}\n'.format(
            date_time_format(config.closing_timestamp))
    smtpserver.sendmail(gmail_user, user.email, header + body)


def send_password_mails(password_f_name, port):
    passwords = pd.read_csv(password_f_name)

    for _, u in passwords.iterrows():
        send_password_mail(u['name'], u['password'], port)


def setup_workflows():
    """Setting up database. 

    Should be called once although there is no harm recalling it: if the
    elements are in the db, it skips adding them.
    """
    extension_names = ['py', 'R', 'txt', 'csv']
    for name in extension_names:
        add_extension(name)

    submission_file_types = [
        ('code', True, 10 ** 5),
        ('text', True, 10 ** 5),
        ('data', False, 10 ** 8)
    ]
    for name, is_editable, max_size in submission_file_types:
        add_submission_file_type(name, is_editable, max_size)

    submission_file_type_extensions = [
        ('code', 'py'),
        ('code', 'R'),
        ('text', 'txt'),
        ('data', 'csv')
    ]
    for type_name, extension_name in submission_file_type_extensions:
        add_submission_file_type_extension(type_name, extension_name)

    workflow_element_types = [
        ('feature_extractor', 'code'),
        ('ts_feature_extractor', 'code'),
        ('imputer', 'code'),
        ('classifier', 'code'),
        ('regressor', 'code'),
        ('calibrator', 'code'),
        ('comments', 'text'),
        ('external_data', 'data'),
    ]
    for workflow_element_type_name, submission_file_type_name in\
            workflow_element_types:
        add_workflow_element_type(
            workflow_element_type_name, submission_file_type_name)

    workflows = [
        ('classifier_workflow', ['classifier']),
        ('regressor_workflow', ['regressor']),
    ]
    for name, element_type_names in workflows:
        add_workflow(name, element_type_names)


def add_extension(name):
    """Adding a new extension, e.g., 'py'."""
    extension = Extension.query.filter_by(name=name).one_or_none()
    if extension is None:
        db.session.add(Extension(name=name))
        db.session.commit()


def add_submission_file_type(name, is_editable, max_size):
    """Adding a new submission file type, e.g., ('code', True, 10 ** 5).

    Should be preceded by adding extensions.
    """
    submission_file_type = SubmissionFileType.query.filter_by(
        name=name).one_or_none()
    if submission_file_type is None:
        db.session.add(SubmissionFileType(
            name=name, is_editable=is_editable, max_size=max_size))
        db.session.commit()


def add_submission_file_type_extension(type_name, extension_name):
    """Adding a new submission file type extension, e.g., ('code', 'py').

    Should be preceded by adding submission file types and extensions.
    """
    submission_file_type = SubmissionFileType.query.filter_by(
        name=type_name).one()
    extension = Extension.query.filter_by(name=extension_name).one()
    type_extension = SubmissionFileTypeExtension.query.filter_by(
        type=submission_file_type, extension=extension).one_or_none()
    if type_extension is None:
        db.session.add(SubmissionFileTypeExtension(
            type=submission_file_type, extension=extension))
        db.session.commit()


def add_workflow_element_type(workflow_element_type_name,
                              submission_file_type_name):
    """Adding a new workflow element type, e.g., ('classifier', 'code').

    Should be preceded by adding submission file types and extensions.
    """
    submission_file_type = SubmissionFileType.query.filter_by(
        name=submission_file_type_name).one()
    workflow_element_type = WorkflowElementType.query.filter_by(
        name=workflow_element_type_name).one_or_none()
    if workflow_element_type is None:
        db.session.add(WorkflowElementType(
            name=workflow_element_type_name, type=submission_file_type))
        db.session.commit()


def add_workflow(workflow_name, element_type_names):
    """Adding a new workflow, e.g., ('classifier_workflow', ['classifier']).

    Workflow file should be set up in
    databoard/specific/workflows/<workflow_name>. Should be preceded by adding
    workflow element types.
    """
    workflow = Workflow.query.filter_by(name=workflow_name).one_or_none()
    if workflow is None:
        db.session.add(Workflow(name=workflow_name))
        workflow = Workflow.query.filter_by(name=workflow_name).one()
        for element_type_name in element_type_names:
            workflow_element_type = WorkflowElementType.query.filter_by(
                name=element_type_name).one()
            workflow_element =\
                WorkflowElement.query.filter_by(
                    workflow=workflow,
                    workflow_element_type=workflow_element_type).one_or_none()
            if workflow_element is None:
                db.session.add(WorkflowElement(
                    workflow=workflow,
                    workflow_element_type=workflow_element_type))
        db.session.commit()


def add_problem(problem_name):
    """Adding a new RAMP problem.

    Problem file should be set up in
    databoard/specific/problems/<problem_name>. Should be preceded by adding
    a workflow, then workflow_name specified in the event file (workflow_name
    is acting as a pointer for the join). Also prepares the data.
    """
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    if problem is None:
        db.session.add(Problem(name=problem_name))
        db.session.commit()
        problem = Problem.query.filter_by(name=problem_name).one()
    problem.module.prepare_data()


def add_event(event_name):
    """Adding a new RAMP event.

    Event file should be set up in
    databoard/specific/events/<event_name>. Should be preceded by adding
    a problem, then problem_name imported in the event file (problem_name
    is acting as a pointer for the join). Also adds CV folds.
    """
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is None:
        db.session.add(Event(name=event_name))
        db.session.commit()
        event = Event.query.filter_by(name=event_name).one()
    _, y_train = event.problem.module.get_train_data()
    cv = event.module.get_cv(y_train)
    for train_is, test_is in cv:
        cv_fold = CVFold(event=event, train_is=train_is, test_is=test_is)
        db.session.add(cv_fold)
    db.session.commit()


def print_events():
    print('***************** Events ****************')
    for i, event in enumerate(Event.query.all()):
        print i, event


def print_problems():
    print('***************** Problems ****************')
    for i, problem in enumerate(Problem.query.all()):
        print i, problem


def print_cv_folds():
    print('***************** CV folds ****************')
    for i, cv_fold in enumerate(CVFold.query.all()):
        print i, cv_fold


def create_user(name, password, lastname, firstname, email,
                access_level='user', hidden_notes=''):
    hashed_password = get_hashed_password(password)
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=email,
                access_level=access_level, hidden_notes=hidden_notes)
    # Creating default team with the same name as the user
    # user is admin of her own team
    team = Team(name=name, admin=user)
    db.session.add(team)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        message = ''
        try:
            User.query.filter_by(name=name).one()
            message += 'username is already in use'
        except NoResultFound:
            # We only check for team names if username is not in db
            try:
                Team.query.filter_by(name=name).one()
                message += 'username is already in use as a team name'
            except NoResultFound:
                pass
        try:
            User.query.filter_by(email=email).one()
            if len(message) > 0:
                message += ' and '
            message += 'email is already in use'
        except NoResultFound:
            pass
        if len(message) > 0:
            raise NameClashError(message)
        else:
            raise e
    logger.info('Creating {}'.format(user))
    logger.info('Creating {}'.format(team))
    return user


def validate_user(user):
    # from 'asked' to 'user'
    user.access_level = 'user'


def get_sandbox(event, user):
    event_team = get_active_user_event_team(event, user)

    submission = Submission.query.filter_by(
        event_team=event_team, name=config.sandbox_d_name).one_or_none()
    return submission


def print_users():
    print('***************** List of users ****************')
    for user in User.query.order_by(User.id):
        print('{} belongs to teams:'.format(user))
        for team in get_user_teams(user):
            print('\t{}'.format(team))
        # print('Sandbox = {}'.format(get_sandbox(event, user)))


######################### Teams ###########################

# for now this is not functional, we should think through how teams
# should be merged when we have multiple RAMP events.
# def merge_teams(name, initiator_name, acceptor_name):
#     initiator = Team.query.filter_by(name=initiator_name).one()
#     acceptor = Team.query.filter_by(name=acceptor_name).one()
#     if not initiator.is_active:
#         raise MergeTeamError('Merge initiator is not active')
#     if not acceptor.is_active:
#         raise MergeTeamError('Merge acceptor is not active')
#     # Meaning they are already in another team for the ramp

#     # Testing if team size is <= max_members_per_team
#     n_members_initiator = get_n_team_members(initiator)
#     n_members_acceptor = get_n_team_members(acceptor)
#     n_members_new = n_members_initiator + n_members_acceptor
#     if n_members_new > config.max_members_per_team:
#         raise MergeTeamError(
#             'Too big team: new team would be of size {}, the max is {}'.format(
#                 n_members_new, config.max_members_per_team))

#     members_initiator = get_team_members(initiator)
#     members_acceptor = get_team_members(acceptor)

#     # Testing if team (same members) exists under a different name. If the
#     # name is the same, we break. If the loop goes through, we add new team.
#     members_set = set(members_initiator).union(set(members_acceptor))
#     for team in Team.query:
#         if members_set == set(get_team_members(team)):
#             if name == team.name:
#                 # ok, but don't add new team, just set them to inactive
#                 initiator.is_active = False
#                 acceptor.is_active = False
#                 return team
#             raise MergeTeamError(
#                 'Team exists with the same members, team name = {}'.format(
#                     team.name))
#     else:
#         team = Team(name=name, admin=initiator.admin,
#                     initiator=initiator, acceptor=acceptor)
#         db.session.add(team)
#     initiator.is_active = False
#     acceptor.is_active = False
#     try:
#         db.session.commit()
#     except IntegrityError as e:
#         db.session.rollback()
#         try:
#             Team.query.filter_by(name=name).one()
#             raise NameClashError('team name is already in use')
#         except NoResultFound:
#             raise e
#     logger.info('Merging {} and {} into {}'.format(initiator, acceptor, team))
#     # submitting the starting kit for merged team
#     make_submission_and_copy_files(
#         name, config.sandbox_d_name, config.sandbox_path)
#     return team


def print_active_teams(event_name):
    print('***************** List of active teams ****************')
    event = Event.query.filter_by(name=event_name).one()
    event_teams = EventTeam.query.filter_by(event=event, is_active=True).all()
    for event_team in event_teams:
        print('{}/{} members:'.format(event_team.event, event_team.team))
        for member in get_team_members(event_team.team):
            print('\t{}'.format(member))


def signup_team(event_name, team_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam(event=event, team=team)
    db.session.add(event_team)
    db.session.commit()
    # submitting the starting kit for team
    from_submission_path = os.path.join(
        config.ramps_path, event.problem.name, config.sandbox_d_name)
    make_submission_and_copy_files(
        event_name, team_name, config.sandbox_d_name, from_submission_path)


######################### Submissions ###########################

def set_state(event_name, team_name, submission_name, state):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one()
    submission.set_state(state)
    db.session.commit()


def delete_submission(event_name, team_name, submission_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam(event=event, team=team)
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one()
    shutil.rmtree(submission.path)
    db.session.delete(submission)
    db.session.commit()


def make_submission_on_cv_folds(cv_folds, submission):
    for cv_fold in cv_folds:
        submission_on_cv_fold = SubmissionOnCVFold(
            submission=submission, cv_fold=cv_fold)
        db.session.add(submission_on_cv_fold)


def make_submission(event_name, team_name, submission_name, submission_path):
    # TODO: to call unit tests on submitted files. Those that are found
    # in the table that describes the workflow. For the rest just check
    # maybe size
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one_or_none()
    if submission is None:
        # Checking if submission is too early
        all_submissions = Submission.query.filter_by(
            event_team=event_team).order_by(
                Submission.submission_timestamp).all()
        if len(all_submissions) == 0:
            last_submission = None
        else:
            last_submission = all_submissions[-1]
        if last_submission is not None and\
                last_submission.name != config.sandbox_d_name:
            now = datetime.datetime.utcnow()
            last = last_submission.submission_timestamp
            min_diff = event.min_duration_between_submissions
            diff = (now - last).total_seconds()
            if diff < min_diff:
                raise TooEarlySubmissionError(
                    'You need to wait {} more seconds until next submission'
                    .format(int(min_diff - diff)))
        submission = Submission(name=submission_name, event_team=event_team)
        db.session.add(submission)
    else:
        # We allow resubmit for new or failing submissions
        if submission.name != config.sandbox_d_name and\
                (submission.state == 'new' or submission.is_error):
            submission.state = 'new'
            submission.submission_timestamp = datetime.datetime.utcnow()
            for submission_on_cv_fold in submission.on_cv_folds:
                # I couldn't figure out how to reset to default values
                db.session.delete(submission_on_cv_fold)
        else:
            raise DuplicateSubmissionError(
                'Submission "{}" of team "{}" at event "{}" exists already'.format(
                    submission_name, team_name, event_name))

    deposited_f_name_list = os.listdir(submission_path)
    # TODO: more error checking
    deposited_types = [f_name.split('.')[0]
                       for f_name in deposited_f_name_list]
    deposited_extensions = [f_name.split('.')[1]
                            for f_name in deposited_f_name_list]
    for workflow_element in event.problem.workflow.elements:
        # We find all files with matching names to workflow_element.name.
        # If none found, raise error.
        # Then look for one that has a legal extension. If none found,
        # raise error. If there are several ones, for now we use the first
        # matching file.

        i_names = [i for i in range(len(deposited_types))
                   if deposited_types[i] == workflow_element.name]
        if len(i_names) == 0:
            db.session.rollback()
            raise MissingSubmissionFileError('{}/{}/{}/{}: {}'.format(
                event_name, team_name, submission_name, workflow_element.name,
                submission_path))

        for i_name in i_names:
            extension_name = deposited_extensions[i_name]
            extension = Extension.query.filter_by(
                name=extension_name).one_or_none()
            if extension is not None:
                break
        else:
            db.session.rollback()
            extensions = [deposited_extensions[i_name] for i_name in i_names]
            extensions = extensions.join(",")
            for i_name in i_names:
                extensions.append(deposited_extensions[i_name])
            raise MissingExtensionError('{}/{}/{}/{}/{}: {}'.format(
                event_name, team_name, submission_name, workflow_element.name,
                extensions, submission_path))

        # maybe it's a resubmit
        submission_file = SubmissionFile.query.filter_by(
            workflow_element=workflow_element,
            submission=submission).one_or_none()
        # TODO: handle if resubmitted file changed extension
        if submission_file is None:
            submission_file_type = SubmissionFileType.query.filter_by(
                name=workflow_element.file_type).one()
            type_extension = SubmissionFileTypeExtension.query.filter_by(
                type=submission_file_type, extension=extension).one()
            submission_file = SubmissionFile(
                submission=submission, workflow_element=workflow_element,
                submission_file_type_extension=type_extension)
#            submission_file = SubmissionFile(
#                submission=submission, name=name,
#                extension_name=extension_name)
            db.session.add(submission_file)

    db.session.commit()  # to enact db.session.delete(submission_on_cv_fold)
    make_submission_on_cv_folds(event.cv_folds, submission)
    # for remembering it in the sandbox view
    event_team.last_submission_name = submission_name
    db.session.commit()

    # We should copy files here
    return submission


def make_submission_and_copy_files(event_name, team_name, new_submission_name,
                                   from_submission_path):
    """Called from signup_team(), merge_teams(), fetch.add_models(),
    view.sandbox()."""

    submission = make_submission(
        event_name, team_name, new_submission_name, from_submission_path)
    # clean up the model directory in case it's a resubmission
    if os.path.exists(submission.path):
        shutil.rmtree(submission.path)
    os.mkdir(submission.path)
    open(os.path.join(submission.path, '__init__.py'), 'a').close()

    # copy the submission files into the model directory, should all this
    # probably go to Submission
    for f_name in submission.f_names:
        src = os.path.join(from_submission_path, f_name)
        dst = os.path.join(submission.path, f_name)
        shutil.copy2(src, dst)  # copying also metadata
        logger.info('Copying {} to {}'.format(src, dst))

    logger.info("Adding submission={}".format(submission))
    return submission


def send_trained_mails(submission):
    if submission.event_team.event.is_send_trained_mails:
        try:
            users = get_team_members(submission.event_team.team)
            recipient_list = [user.email for user in users]
            gmail_user = config.MAIL_USERNAME
            gmail_pwd = config.MAIL_PASSWORD
            smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
            smtpserver.ehlo()
            smtpserver.starttls()
            smtpserver.ehlo
            smtpserver.login(gmail_user, gmail_pwd)
            subject = 'submission {} for {} was trained at {}'.format(
                submission.name, submission.event_team.event.name, 
                date_time_format(submission.training_timestamp))
            header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
                recipient_list, gmail_user, subject)
            body = ''
            for recipient in recipient_list:
                smtpserver.sendmail(gmail_user, recipient, header + body)
        except Exception as e:
            logger.error('Mailing error: {}'.format(e))


def train_test_submissions(submissions=None, force_retrain_test=False):
    """Trains and tests submission.

    If submissions is None, trains and tests all submissions.
    """

    if submissions is None:
        submissions = Submission.query.filter(
            Submission.name != 'sandbox').order_by(Submission.id).all()
    for submission in submissions:
        train_test_submission(submission, force_retrain_test)


# For parallel call
def train_test_submission(submission, force_retrain_test=False):
    """We do it here so it's dockerizable."""

    detached_submission_on_cv_folds = [
        DetachedSubmissionOnCVFold(submission_on_cv_fold)
        for submission_on_cv_fold in submission.on_cv_folds]

    if force_retrain_test:
        logger.info('Forced retraining/testing {}'.format(submission))

    X_train, y_train = submission.event.problem.module.get_train_data()
    X_test, y_test = submission.event.problem.module.get_test_data()

    # Parallel, dict
    if config.is_parallelize:
        # We are using 'threading' so train_test_submission_on_cv_fold
        # updates the detached submission_on_cv_fold objects. If it doesn't
        # work, we can go back to multiprocessing and
        logger.info('Number of processes = {}'.format(
            submission.event.n_CV))
        detached_submission_on_cv_folds = Parallel(
            n_jobs=submission.event.n_CV, verbose=5)(
            delayed(train_test_submission_on_cv_fold)(
                submission_on_cv_fold, X_train, y_train, X_test, y_test,
                force_retrain_test)
            for submission_on_cv_fold in detached_submission_on_cv_folds)
        for detached_submission_on_cv_fold, submission_on_cv_fold in\
                zip(detached_submission_on_cv_folds, submission.on_cv_folds):
            submission_on_cv_fold.update(detached_submission_on_cv_fold)
    else:
        # detached_submission_on_cv_folds = []
        for detached_submission_on_cv_fold, submission_on_cv_fold in\
                zip(detached_submission_on_cv_folds, submission.on_cv_folds):
            train_test_submission_on_cv_fold(
                detached_submission_on_cv_fold,
                X_train, y_train, X_test, y_test,
                force_retrain_test)
            submission_on_cv_fold.update(detached_submission_on_cv_fold)
    submission.training_timestamp = datetime.datetime.utcnow()
    submission.set_state_after_training()
    submission.compute_test_score_cv_bag()
    submission.compute_valid_score_cv_bag()
    db.session.commit()
    logger.info('valid_score = {}'.format(submission.valid_score_cv_bag))
    logger.info('test_score = {}'.format(submission.test_score_cv_bag))

    send_trained_mails(submission)


def _make_error_message(e):
    """log_msg is the full error what we print into logger.error. error_msg
    is what we save and display to the user. Ideally error_msg is the part
    of the code that is related to the user submission.
    """
    if hasattr(e, 'traceback'):
        log_msg = str(e.traceback)
    else:
        log_msg = repr(e)
    error_msg = log_msg
    # TODO: It the user calls something in his classifier, that part of the
    # stack is lost. We should make this more intelligent.
    cut_exception_text = error_msg.rfind('--->')
    if cut_exception_text > 0:
        error_msg = error_msg[cut_exception_text:]
    return log_msg, error_msg


def train_test_submission_on_cv_fold(detached_submission_on_cv_fold, 
                                     X_train, y_train,
                                     X_test, y_test, force_retrain_test=False):
    train_submission_on_cv_fold(
        detached_submission_on_cv_fold, X_train, y_train,
        force_retrain=force_retrain_test)
    if 'error' not in detached_submission_on_cv_fold.state:
        test_submission_on_cv_fold(
            detached_submission_on_cv_fold, X_test, y_test,
            force_retest=force_retrain_test)
    # When called in a single thread, we don't need the return value,
    # submission_on_cv_fold is modified in place. When called in parallel
    # multiprocessing mode, however, copies are made when the function is
    # called, so we have to explicitly return the modified object (so it is
    # ercopied into the original object)
    return detached_submission_on_cv_fold


def train_submission_on_cv_fold(detached_submission_on_cv_fold, X, y,
                                force_retrain=False):
    if detached_submission_on_cv_fold.state not in ['new', 'checked']\
            and not force_retrain:
        if 'error' in detached_submission_on_cv_fold.state:
            logger.error('Trying to train failed {}'.format(
                detached_submission_on_cv_fold))
        else:
            logger.info('Already trained {}'.format(
                detached_submission_on_cv_fold))
        return

    # so to make it importable, TODO: should go to make_submission
    # open(os.path.join(self.submission.path, '__init__.py'), 'a').close()

    train_is = detached_submission_on_cv_fold.train_is

    logger.info('Training {}'.format(detached_submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        detached_submission_on_cv_fold.trained_submission =\
            detached_submission_on_cv_fold.train_submission(
                detached_submission_on_cv_fold.module, X, y, train_is)
        detached_submission_on_cv_fold.state = 'trained'
    except Exception, e:
        detached_submission_on_cv_fold.state = 'training_error'
        log_msg, detached_submission_on_cv_fold.error_msg =\
            _make_error_message(e)
        logger.error(
            'Training {} failed with exception: \n{}'.format(
                detached_submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    detached_submission_on_cv_fold.train_time = end - start

    logger.info('Validating {}'.format(detached_submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        y_pred = detached_submission_on_cv_fold.test_submission(
            detached_submission_on_cv_fold.trained_submission, X,
            range(len(y)))
        if len(y_pred) == len(y):
            detached_submission_on_cv_fold.full_train_y_pred = y_pred
            detached_submission_on_cv_fold.state = 'validated'
        else:
            detached_submission_on_cv_fold.error_msg =\
                'Wrong output dimension in ' +\
                'predict: {} instead of {}'.format(len(y_pred), len(y))
            detached_submission_on_cv_fold.state = 'validating_error'
            logger.error(
                'Validating {} failed with exception: \n{}'.format(
                    detached_submission_on_cv_fold.error_msg))
            return
    except Exception, e:
        detached_submission_on_cv_fold.state = 'validating_error'
        log_msg, detached_submission_on_cv_fold.error_msg =\
            _make_error_message(e)
        logger.error(
            'Validating {} failed with exception: \n{}'.format(
                detached_submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    detached_submission_on_cv_fold.valid_time = end - start


def test_submission_on_cv_fold(detached_submission_on_cv_fold, X, y,
                               force_retest=False):
    if detached_submission_on_cv_fold.state not in\
            ['new', 'checked', 'trained', 'validated'] and not force_retest:
        if 'error' in detached_submission_on_cv_fold.state:
            logger.error('Trying to test failed {}'.format(
                detached_submission_on_cv_fold))
        else:
            logger.info('Already tested {}'.format(detached_submission_on_cv_fold))
        return

    logger.info('Testing {}'.format(detached_submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        y_pred = detached_submission_on_cv_fold.test_submission(
            detached_submission_on_cv_fold.trained_submission, X, range(len(y)))
        if len(y_pred) == len(y):
            detached_submission_on_cv_fold.test_y_pred = y_pred
            detached_submission_on_cv_fold.state = 'tested'
        else:
            detached_submission_on_cv_fold.error_msg =\
                'Wrong output dimension in ' +\
                'predict: {} instead of {}'.format(len(y_pred), len(y))
            detached_submission_on_cv_fold.state = 'testing_error'
            logger.error(
                'Testing {} failed with exception: \n{}'.format(
                    detached_submission_on_cv_fold.error_msg))
    except Exception, e:
        detached_submission_on_cv_fold.state = 'testing_error'
        log_msg, detached_submission_on_cv_fold.error_msg = _make_error_message(e)
        logger.error(
            'Testing {} failed with exception: \n{}'.format(
                detached_submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    detached_submission_on_cv_fold.test_time = end - start


def compute_contributivity(event_name, force_ensemble=False):
    """Computes contributivity leaderboard scores.

    Parameters
    ----------
    event_name : string
    force_ensemble : boolean
        To force include deleted models.
    """
    logger.info('Combining models')
    # The following should go into config, we'll get there when we have a
    # lot of models.
    # One of Caruana's trick: bag the models
    # selected_index_lists = np.array([random.sample(
    #    range(len(models_df)), int(0.8*models_df.shape[0]))
    #    for _ in range(n_bags)])
    # Or you can select a subset
    # selected_index_lists = np.array([[24, 26, 28, 31]])
    # Or just take everybody
    # Now all of this should be handled by submission.is_to_ensemble parameter
    # It would also make more sense to bag differently in eah fold, see the
    # comment in cv_fold.get_combined_predictions

    event = Event.query.filter_by(name=event_name).one()
    submissions = get_submissions(event_name)

    true_predictions_train = event.problem.true_predictions_train()
    true_predictions_test = event.problem.true_predictions_test()

    combined_predictions_list = []
    best_predictions_list = []
    combined_test_predictions_list = []
    best_test_predictions_list = []
    test_is_list = []
    for cv_fold in CVFold.query.filter_by(event=event).all():
        logger.info('{}'.format(cv_fold))
        true_predictions_valid = event.problem.true_predictions_valid(
            cv_fold.test_is)
        combined_predictions, best_predictions,\
            combined_test_predictions, best_test_predictions =\
            _compute_contributivity_on_fold(
                cv_fold, true_predictions_valid, force_ensemble)
        # TODO: if we do asynchron CVs, this has to be revisited
        if combined_predictions is None:
            logger.info('No submissions to combine')
            return
        combined_predictions_list.append(combined_predictions)
        best_predictions_list.append(best_predictions)
        combined_test_predictions_list.append(combined_test_predictions)
        best_test_predictions_list.append(best_test_predictions)
        test_is_list.append(cv_fold.test_is)
    for submission in submissions:
        submission.set_contributivity(is_commit=False)
    db.session.commit()
    from model import _get_score_cv_bags
    # if there are no predictions to combine, it crashed
    combined_predictions_list = [c for c in combined_predictions_list
                                 if c is not None]
    if len(combined_predictions_list) > 0:
        scores = _get_score_cv_bags(
            event, combined_predictions_list, true_predictions_train,
            test_is_list)
        logger.info('Combined combined valid score = {}'.format(scores))
        with open('combined_combined_valid_score.txt', 'w') as f:
            f.write(str(round(scores[-1], config.score_precision)))
            # TODO: should go into db
    else:
        with open('combined_combined_valid_score.txt', 'w') as f:
            f.write('')  # TODO: should go into db

    best_predictions_list = [c for c in best_predictions_list
                             if c is not None]
    if len(best_predictions_list) > 0:
        logger.info('Combined foldwise best valid score = {}'.format(
            _get_score_cv_bags(
                event, best_predictions_list, true_predictions_train, 
                test_is_list)))

    combined_test_predictions_list = [c for c in combined_test_predictions_list
                                      if c is not None]
    if len(combined_test_predictions_list) > 0:
        scores = _get_score_cv_bags(
            event, combined_test_predictions_list, true_predictions_test)
        logger.info('Combined combined test score = {}'.format(scores))
        with open('combined_combined_test_score.txt', 'w') as f:
            f.write(str(round(scores[-1], config.score_precision)))
    else:
        with open('combined_combined_test_score.txt', 'w') as f:
            f.write('')  # TODO: should go into db

    best_test_predictions_list = [c for c in best_test_predictions_list
                                  if c is not None]
    if len(best_test_predictions_list) > 0:
        logger.info('Combined foldwise best test score = {}'.format(
            _get_score_cv_bags(
                event, best_test_predictions_list, true_predictions_test)))


def _compute_contributivity_on_fold(cv_fold, true_predictions_valid,
                                    force_ensemble=False):
    """Construct the best model combination on a single fold.

    Using greedy forward selection with replacement. See
    http://www.cs.cornell.edu/~caruana/ctp/ct.papers/
    caruana.icml04.icdm06long.pdf.
    Then sets foldwise contributivity.

    Parameters
    ----------
    force_ensemble : boolean
        To force include deleted models
    """
    # The submissions must have is_to_ensemble set to True. It is for
    # fogetting models. Users can also delete models in which case
    # we make is_valid false. We then only use these models if
    # force_ensemble is True.
    # We can further bag here which should be handled in config (or
    # ramp table.) Or we could bag in get_next_best_single_fold

    # this is the bottleneck
    selected_submissions_on_fold = [
        submission_on_fold for submission_on_fold in cv_fold.submissions
        if (submission_on_fold.submission.is_valid or force_ensemble)
        and submission_on_fold.submission.is_to_ensemble
        and submission_on_fold.is_public_leaderboard
        and submission_on_fold.submission.name != config.sandbox_d_name
    ]
    if len(selected_submissions_on_fold) == 0:
        return None, None, None, None
    # TODO: maybe this can be simplified. Don't need to get down
    # to prediction level.
    predictions_list = [
        submission_on_fold.valid_predictions
        for submission_on_fold in selected_submissions_on_fold]
    valid_scores = [
        # To convert it into Score instance which knows if lower/heigher
        # is better
        cv_fold.event.score.convert(submission_on_fold.valid_score)
        for submission_on_fold in selected_submissions_on_fold]
    best_prediction_index = np.argmax(valid_scores)
    best_index_list = np.array([best_prediction_index])
    improvement = True
    while improvement and len(best_index_list) < config.max_n_ensemble:
        old_best_index_list = best_index_list
        best_index_list, score = get_next_best_single_fold(
            cv_fold.event, predictions_list, true_predictions_valid,
            best_index_list)
        improvement = len(best_index_list) != len(old_best_index_list)
        logger.info('\t{}: {}'.format(best_index_list, score))
    # reset
    for submission_on_fold in selected_submissions_on_fold:
        submission_on_fold.best = False
        submission_on_fold.contributivity = 0.0
    # set
    selected_submissions_on_fold[best_index_list[0]].best = True
    # we share a unit of 1. among the contributive submissions
    unit_contributivity = 1. / len(best_index_list)
    for i in best_index_list:
        selected_submissions_on_fold[i].contributivity +=\
            unit_contributivity
    combined_predictions = combine_predictions_list(
        predictions_list, index_list=best_index_list)
    best_predictions = predictions_list[best_index_list[0]]

    test_predictions_list = [
        submission_on_fold.test_predictions
        for submission_on_fold in selected_submissions_on_fold
    ]
    if any(test_predictions_list) is None:
        logger.error("Can't compute combined test score," +
                     " some submissions are untested.")
        combined_test_predictions = None
        best_test_predictions = None
    else:
        combined_test_predictions = combine_predictions_list(
            test_predictions_list, index_list=best_index_list)
        best_test_predictions = test_predictions_list[best_index_list[0]]

    return combined_predictions, best_predictions,\
        combined_test_predictions, best_test_predictions


def get_public_leaderboard(event_name, team_name=None, user_name=None, 
                           is_open_code=True):
    """
    Returns
    -------
    leaderboard_html : html string
    """

    submissions = get_submissions(event_name, team_name, user_name)
    submissions = [submission for submission in submissions
                   if submission.is_public_leaderboard]

    columns = ['team',
               'submission',
               'score',
               'contributivity',
               'train time',
               'test time',
               'submitted at (UTC)']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [submission.event_team.team.name,
                      submission.name_with_link if is_open_code
                      else submission.name[:20],
                      round(submission.valid_score_cv_bag,
                            submission.event_team.event.score_precision),
                      int(100 * submission.contributivity + 0.5),
                      int(submission.train_time_cv_mean + 0.5),
                      int(submission.valid_time_cv_mean + 0.5),
                      date_time_format(submission.submission_timestamp)])}
        for submission in submissions
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    leaderboard_df = leaderboard_df.sort_values(
        'contributivity', ascending=False)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)
    return leaderboard_html


def get_private_leaderboard(event_name, team_name=None, user_name=None):
    """
    Returns
    -------
    leaderboard_html : html string
    """

    submissions = get_submissions(event_name, team_name, user_name)
    submissions = [submission for submission in submissions
                   if submission.is_private_leaderboard]

    columns = ['team',
               'submission',
               'public bag',
               'public mean',
               'public std',
               'private bag',
               'private mean',
               'private std',
               'contributivity',
               'train time',
               'trt std',
               'test time',
               'tet std',
               'submitted at (UTC)']
    precision = submission.event_team.event.score_precision
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [submission.event_team.team.name,
                      submission.name_with_link,
                      round(submission.valid_score_cv_bag, precision),
                      round(submission.valid_score_cv_mean, precision),
                      round(submission.valid_score_cv_std, precision + 1),
                      round(submission.test_score_cv_bag, precision),
                      round(submission.test_score_cv_mean, precision),
                      round(submission.test_score_cv_std, precision + 1),
                      int(100 * submission.contributivity + 0.5),
                      round(submission.train_time_cv_mean),
                      round(submission.train_time_cv_std),
                      round(submission.valid_time_cv_mean),
                      round(submission.valid_time_cv_std),
                      date_time_format(submission.submission_timestamp)])}
        for submission in submissions
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    leaderboard_df = leaderboard_df.sort_values(
        'contributivity', ascending=False)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)
    return leaderboard_html


def get_failed_leaderboard(event_name, team_name=None, user_name=None):
    """
    Returns
    -------
    leaderboard_html : html string
    """
    submissions = get_submissions(event_name, team_name, user_name)
    submissions = [submission for submission in submissions
                   if submission.is_error]

    columns = ['team',
               'submission',
               'submitted at (UTC)',
               'error']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [submission.event_team.team.name,
                      submission.name_with_link,
                      date_time_format(submission.submission_timestamp),
                      submission.state_with_link])}
        for submission in submissions
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)
    return leaderboard_html


def get_new_leaderboard(event_name, team_name=None, user_name=None):
    """
    Returns
    -------
    leaderboard_html : html string
    """
    submissions = get_submissions(event_name, team_name, user_name)
    submissions = [submission for submission in submissions
                   if submission.state == 'new' and submission.is_not_sandbox]

    columns = ['team',
               'submission',
               'submitted at (UTC)']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [submission.event_team.team.name,
                      submission.name_with_link,
                      date_time_format(submission.submission_timestamp)])}
        for submission in submissions
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)
    return leaderboard_html


def get_submissions(event_name=None, user_name=None, team_name=None, 
                    submission_name=None):
    if event_name is None:  # All submissions
        submissions = Submission.query.all()
    else:
        if team_name is None: 
            if user_name is None:  # All submissions in a given event
                submissions_ = db.session.query(
                    Submission, Event, EventTeam).filter(
                    Event.name == event_name).filter(
                    Event.id == EventTeam.event_id).filter(
                    EventTeam.id == Submission.event_team_id).all()
            else:
                # All submissions for a given event by all the teams of a user
                submissions_ = []
                user_teams = get_user_event_teams(event_name, user_name)
                for team in user_teams:
                    submissions_ += db.session.query(
                        Submission, Event, Team, EventTeam).filter(
                        Team.id == team.id).filter(
                        Team.id == Submission.team_id).filter(
                        Event.name == event_name).filter(
                        Event.id == EventTeam.event_id).filter(
                        EventTeam.id == Submission.event_team_id).all()
        else:
            if submission_name is None:
                # All submissions in a given event and team
                submissions_ = db.session.query(
                    Submission, Event, Team, EventTeam).filter(
                    Event.name == event_name).filter(
                    Team.name == team_name).filter(
                    Event.id == EventTeam.event_id).filter(
                    Team.id == EventTeam.team_id).filter(
                    EventTeam.id == Submission.event_team_id).all()
            else:  # Given submission
                submissions_ = db.session.query(
                    Submission, Event, Team, EventTeam).filter(
                    Submission.name == submission_name).filter(
                    Event.name == event_name).filter(
                    Team.name == team_name).filter(
                    Event.id == EventTeam.event_id).filter(
                    Team.id == EventTeam.team_id).filter(
                    EventTeam.id == Submission.event_team_id).all()
        submissions = list(zip(*submissions_)[0])
    return submissions


def get_submissions_of_state(state):
    return Submission.query.filter(Submission.state == state).all()


def get_earliest_new_submission():
    new_submissions = Submission.query.filter_by(
        state='new').order_by(
        Submission.submission_timestamp).all()
    if len(new_submissions) == 0:
        return None
    else:
        return new_submissions[0]


def print_submissions(event_name=None, team_name=None, submissions=None):
    print('***************** List of submissions ****************')
    for submission in get_submissions(event_name, team_name, submissions):
        print submission
        print('\tstate = {}'.format(submission.state))
        print('\tcontributivity = {0:.2f}'.format(
            submission.contributivity))
        print('\tvalid_score_cv_mean = {0:.2f}'.format(
            submission.valid_score_cv_mean))
        print '\tvalid_score_cv_bag = {0:.2f}'.format(
            float(submission.valid_score_cv_bag))
        print '\tvalid_score_cv_bags = {}'.format(
            submission.valid_score_cv_bags)
        print '\ttest_score_cv_mean = {0:.2f}'.format(
            submission.test_score_cv_mean)
        print '\ttest_score_cv_bag = {0:.2f}'.format(
            float(submission.test_score_cv_bag))
        print '\ttest_score_cv_bags = {}'.format(
            submission.test_score_cv_bags)
        print('\tpath = {}'.format(submission.path))
        print '\tcv folds'
        submission_on_cv_folds = db.session.query(SubmissionOnCVFold).filter(
            SubmissionOnCVFold.submission == submission).all()
        for submission_on_cv_fold in submission_on_cv_folds:
            print '\t\t' + str(submission_on_cv_fold)


def set_error(team_name, submission_name, error, error_msg):
    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        team=team, name=submission_name).one()
    submission.set_error(error, error_msg)
    db.session.commit()


def get_top_score_of_user(user, closing_timestamp):
    """Returns the bagged test score of the submission with the best bagged
    valid score, from among all the submissions of the user, before 
    closing_timestamp.
    """
    team = get_active_user_team(user)
    submissions = Submission.query.filter_by(team=team).filter(
        Submission.is_private_leaderboard).all()
    best_valid_score = config.config_object.specific.score.zero
    best_test_score = config.config_object.specific.score.zero
    for submission in submissions:
        if submission.valid_score_cv_bag > best_valid_score:
            best_valid_score = submission.valid_score_cv_bag
            best_test_score = submission.test_score_cv_bag
    return best_test_score


def get_top_score_per_user(closing_timestamp=None):
    if closing_timestamp is None:
        closing_timestamp = datetime.datetime.utcnow()
    users = db.session.query(User).all()
    columns = ['name',
               'score']
    top_score_per_user_dict = [
        {column: value for column, value in zip(
            columns, [user.name,
                      get_top_score_of_user(user, closing_timestamp)])
        }
        for user in users
    ]
    top_score_per_user_dict_df = pd.DataFrame(
        top_score_per_user_dict, columns=columns)
    top_score_per_user_dict_df = top_score_per_user_dict_df.sort_values('name')
    return top_score_per_user_dict_df


#################### User interactions #####################

def add_user_interaction(**kwargs):
    user_interaction = UserInteraction(**kwargs)
    db.session.add(user_interaction)
    db.session.commit()


def print_user_interactions():
    print('*********** User interactions ****************')
    for user_interaction in UserInteraction.query.all():
        print date_time_format(user_interaction.timestamp),\
            user_interaction.user, user_interaction.interaction
        if user_interaction.submission_file_diff is not None:
            print user_interaction.submission_file_diff
        if user_interaction.submission_file_similarity is not None:
            print user_interaction.submission_file_similarity
        if user_interaction.submission is not None:
            print user_interaction.submission


def get_user_interactions():
    """
    Returns
    -------
    user_interactions_html : html string
    """

    user_interactions = UserInteraction.query.all()

    columns = ['timestamp (UTC)',
               'user',
               'team',
               'interaction',
               'submission',
               'file',
               'code similarity',
               'diff']

    def submission_name(user_interaction):
        submission = user_interaction.submission
        if submission is None:
            return ''
        else:
            return submission.team.name + '/' + submission.name_with_link

    def submission_file_name(user_interaction):
        submission_file = user_interaction.submission_file
        if submission_file is None:
            return ''
        else:
            return submission_file.name_with_link

    def submission_similarity(user_interaction):
        similarity = user_interaction.submission_file_similarity
        if similarity is None:
            return ''
        else:
            return str(round(similarity, 2))

    def submission_diff_with_link(user_interaction):
        diff_link = user_interaction.submission_file_diff_link
        if diff_link is None:
            return ''
        else:
            return '<a href="' + diff_link + '">diff</a>'

    user_interactions_dict_list = [
        {column: value for column, value in zip(
            columns, [date_time_format(user_interaction.timestamp),
                      user_interaction.user.name,
                      user_interaction.team.name,
                      user_interaction.interaction,
                      submission_name(user_interaction),
                      submission_file_name(user_interaction),
                      submission_similarity(user_interaction),
                      submission_diff_with_link(user_interaction)
                      ])}
        for user_interaction in user_interactions
    ]
    user_interactions_df = pd.DataFrame(
        user_interactions_dict_list, columns=columns)
    user_interactions_df = user_interactions_df.sort_values(
        'timestamp (UTC)', ascending=False)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    user_interactions_html = user_interactions_df.to_html(**html_params)
    return user_interactions_html
