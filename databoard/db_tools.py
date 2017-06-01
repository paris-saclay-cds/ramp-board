import os
import time
import codecs
import shutil
import bcrypt
import timeit
import smtplib
import logging
import datetime
import json
import zlib
import base64
import numpy as np
import pandas as pd
from itertools import chain
import xkcdpass.xkcd_password as xp
from sklearn.externals.joblib import Parallel, delayed
from databoard import db
from sklearn.utils.validation import assert_all_finite
from importlib import import_module

from databoard.model import User, Team, Submission, SubmissionFile,\
    SubmissionFileType, SubmissionFileTypeExtension, WorkflowElementType,\
    WorkflowElement, Workflow, Extension, Problem, Event, EventTeam,\
    ScoreType, EventScoreType, SubmissionSimilarity,\
    CVFold, SubmissionOnCVFold, DetachedSubmissionOnCVFold,\
    UserInteraction, EventAdmin,\
    NameClashError, TooEarlySubmissionError,\
    DuplicateSubmissionError, MissingSubmissionFileError,\
    MissingExtensionError, Keyword, ProblemKeyword,\
    combine_predictions_list, get_next_best_single_fold,\
    get_active_user_event_team, get_user_teams,\
    get_team_members, get_user_event_teams
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

import databoard.config as config
from databoard import post_api
# from databoard import celery

logger = logging.getLogger('databoard')
pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output


def date_time_format(date_time):
    return date_time.strftime('%Y-%m-%d %H:%M:%S %a')


def table_format(table_html):
    """
    remove <table></table> keywords from html table (converted from pandas
    dataframe), to insert in datatable
    """
    # return '<thead> %s </tbody><tfoot><tr></tr></tfoot>' %\
    return '<thead> %s </tbody>' %\
        table_html.split('<thead>')[1].split('</tbody>')[0]


######################### Users ###########################

def get_hashed_password(plain_text_password):
    """Hash a password for the first time.

    (Using bcrypt, the salt is saved into the hash itself)
    """
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hased password.

    Using bcrypt, the salt is saved into the hash itself.
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
    """Send password mail.

    Also resets password. If port is None, use
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
    subject = 'RAMP signup information'
    header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
        user.email, gmail_user, subject.encode('utf-8'))

    body = 'Here is your login and other information ' +\
        'for the RAMP site:\n\n'
    body += 'username: {}\n'.format(user.name.encode('utf-8'))
    body += 'password: {}\n'.format(password)
    body += 'submission site: http://www.ramp.studio'
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


def add_extension(name):
    """Adding a new extension, e.g., 'py'."""
    extension = Extension.query.filter_by(name=name).one_or_none()
    if extension is None:
        extension = Extension(name=name)
        logger.info('Adding {}'.format(extension))
        db.session.add(extension)
        db.session.commit()


def add_submission_file_type(name, is_editable, max_size):
    """Adding a new submission file type, e.g., ('code', True, 10 ** 5).
`
    Should be preceded by adding extensions.
    """
    submission_file_type = SubmissionFileType.query.filter_by(
        name=name).one_or_none()
    if submission_file_type is None:
        submission_file_type = SubmissionFileType(
            name=name, is_editable=is_editable, max_size=max_size)
        logger.info('Adding {}'.format(submission_file_type))
        db.session.add(submission_file_type)
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
        type_extension = SubmissionFileTypeExtension(
            type=submission_file_type, extension=extension)
        logger.info('Adding {}'.format(type_extension))
        db.session.add(type_extension)
        db.session.commit()


def add_workflow(workflow_object):
    """Adding a new workflow.

    Workflow class should exist in rampwf.workflows. The name of the
    workflow will be the classname (e.g. Classifier). Element names 
    are taken from workflow.element_names. Element types are inferred
    from the extension. This is important because e.g. the max size
    and the editability will depend on the type.

<<<<<<< HEAD
    Workflow class should exist in rampwf.workflows. The name of the
    workflow will be the classname (e.g. Classifier). Element names 
    are taken from workflow.element_names. Element types are inferred
    from the extension. This is important because e.g. the max size
    and the editability will depend on the type.
=======
def add_workflow(workflow_object):
    """Adding a new workflow, e.g., ('classifier_workflow', ['classifier']).
>>>>>>> 1) get_cv now complies with sklearn (X and y)

    add_workflow is called by add_problem, taking the workflow to add
    from the problem.py file of the starting kit.
    """
<<<<<<< HEAD
    # name is the name of the workflow *Class*, not the module
=======
>>>>>>> 1) get_cv now complies with sklearn (X and y)
    workflow_name = type(workflow_object).__name__
    workflow_element_names = workflow_object.element_names
    workflow = Workflow.query.filter_by(name=workflow_name).one_or_none()
    if workflow is None:
        db.session.add(Workflow(name=workflow_name))
        workflow = Workflow.query.filter_by(name=workflow_name).one()
    for element_name in workflow_element_names:
<<<<<<< HEAD
        tokens = element_name.split('.')
        element_file_name = tokens[0]
        # inferring that file is code if there is no extension
        element_file_extension_name = 'py'
        if len(tokens) > 1:
            element_file_extension_name = tokens[1]
        if len(tokens) > 2:
            raise ValueError(
                'File name {} should contain at most one "."'.format(
                    element_name))
        extension = Extension.query.filter_by(
            name=element_file_extension_name).one_or_none()
        if extension is None:
            raise ValueError(
                'Unknown extension {}'.format(element_file_extension_name))
        type_extension = SubmissionFileTypeExtension.query.filter_by(
            extension=extension).one_or_none()
        if type_extension is None:
            raise ValueError(
                'Unknown file type {}'.format(element_file_extension_name))

        workflow_element_type = WorkflowElementType.query.filter_by(
            name=element_file_name).one_or_none()
        if workflow_element_type is None:
            workflow_element_type = WorkflowElementType(
                name=element_file_name, type=type_extension.type)
            logger.info('Adding {}'.format(workflow_element_type))
            db.session.add(workflow_element_type)
            db.session.commit()
=======
        workflow_element_type = WorkflowElementType.query.filter_by(
            name=element_name).one()
>>>>>>> 1) get_cv now complies with sklearn (X and y)
        workflow_element =\
            WorkflowElement.query.filter_by(
                workflow=workflow,
                workflow_element_type=workflow_element_type).one_or_none()
        if workflow_element is None:
            workflow_element = WorkflowElement(
                workflow=workflow,
                workflow_element_type=workflow_element_type)
            logger.info('Adding {}'.format(workflow_element))
            db.session.add(workflow_element)
    db.session.commit()

def add_problem(problem_name, force=False):
    """Adding a new RAMP problem.

    Problem file should be set up in
    databoard/specific/problems/<problem_name>. Should be preceded by adding
    a workflow, then workflow_name specified in the event file (workflow_name
    is acting as a pointer for the join). Also prepares the data.
    """
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    if problem is not None:
        if force:
            delete_problem(problem)
        else:
            logger.info(
                'Attempting to delete problem and all linked events, ' +
                'use "force=True" if you know what you are doing.')
            return
    problem_module = import_module('.' + problem_name, config.problems_module)
    add_workflow(problem_module.workflow)
    problem = Problem(name=problem_name)
    logger.info('Adding {}'.format(problem))
    db.session.add(problem)
    db.session.commit()
    problem.module.prepare_data()


# these could go into a delete callback in problem and event, I just don't know
# how to do that.
def delete_problem(problem):
    for event in problem.events:
        delete_event(problem)
    db.session.delete(problem)
    db.session.commit()


# the main reason having this is that I couldn't make a cascade delete in
# SubmissionSimilarity since it has two submission parents
def delete_event(event):
    submissions = get_submissions(event_name=event.name)
    delete_submission_similarity(submissions)
    db.session.delete(event)
    db.session.commit()


# the main reason having this is that I couldn't make a cascade delete in
# SubmissionSimilarity since it has two submission parents
def delete_submission_similarity(submissions):
    submission_similaritys = []
    for submission in submissions:
        submission_similaritys += SubmissionSimilarity.query.filter_by(
            target_submission=submission).all()
        submission_similaritys += SubmissionSimilarity.query.filter_by(
            source_submission=submission).all()
    for submission_similarity in submission_similaritys:
        db.session.delete(submission_similarity)
    db.session.commit()


def _set_table_attribute(table, attr):
    """Setting attributes from config file.

    Assumes that table has a module field that imports the config file.
    If attr is not specified in the file, revert to default.
    """
    try:
        value = getattr(table.module, attr)
    except AttributeError:
        return
    setattr(table, attr, value)


def add_event(event_name, force=False):
    """Adding a new RAMP event.

    Event file should be set up in
    databoard/specific/events/<event_name>. Should be preceded by adding
    a problem, then problem_name imported in the event file (problem_name
    is acting as a pointer for the join). Also adds CV folds.
    """
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is not None:
        if force:
            delete_event(event)
        else:
            logger.info(
                'Attempting to delete event, use "force=True" ' +
                'if you know what you are doing')
            return
    event = Event(name=event_name)
    logger.info('Adding {}'.format(event))
    db.session.add(event)
    db.session.commit()

    _set_table_attribute(event, 'max_members_per_team')
    _set_table_attribute(event, 'max_n_ensemble')
    _set_table_attribute(event, 'score_precision')
    # _set_table_attribute(event, 'n_cv')
    _set_table_attribute(event, 'is_send_trained_mails')
    _set_table_attribute(event, 'is_send_submitted_mails')
    _set_table_attribute(event, 'min_duration_between_submissions')
    _set_table_attribute(event, 'opening_timestamp')
    _set_table_attribute(event, 'public_opening_timestamp')
    _set_table_attribute(event, 'closing_timestamp')
    _set_table_attribute(event, 'is_public')
    _set_table_attribute(event, 'is_controled_signup')

    X_train, y_train = event.problem.module.get_train_data()
    cv = event.module.get_cv(X_train, y_train)
    for train_is, test_is in cv:
        cv_fold = CVFold(event=event, train_is=train_is, test_is=test_is)
        db.session.add(cv_fold)

    score_types = event.module.score_types
    for score_type in score_types:
        event_score_type = EventScoreType(
            event=event, score_type_object=score_type)
        db.session.add(event_score_type)
    event.official_score_name = score_types[0].name
<<<<<<< HEAD
    db.session.commit()


def add_keyword(name, type, category=None, description=None, force=False):
    keyword = Keyword.query.filter_by(name=name).one_or_none()
    if keyword is not None:
        if force:
            keyword.type = type
            keyword.category = category
            keyword.description = description
        else:
            logger.info(
                'Attempting to update existing keyword, use ' +
                '"force=True" if you really want to update.')
            return
    else:
        keyword = Keyword(
            name=name, type=type, category=category, description=description)
        db.session.add(keyword)
    db.session.commit()


def add_problem_keyword(
        problem_name, keyword_name, description=None, force=False):
    problem = Problem.query.filter_by(name=problem_name).one()
    keyword = Keyword.query.filter_by(name=keyword_name).one()
    problem_keyword = ProblemKeyword.query.filter_by(
        problem=problem, keyword=keyword).one_or_none()
    if problem_keyword is not None:
        if force:
            problem_keyword.description = description
        else:
            logger.info(
                'Attempting to update existing problem-keyword, use ' +
                '"force=True" if you really want to update.')
            return
    else:
        problem_keyword = ProblemKeyword(
            problem=problem, keyword=keyword, description=description)
        db.session.add(problem_keyword)
    db.session.commit()


def print_events():
    print('***************** Events ****************')
    for i, event in enumerate(Event.query.all()):
        print(i, event)


def print_problems():
    print('***************** Problems ****************')
    for i, problem in enumerate(Problem.query.all()):
        print(i, problem)


def print_cv_folds():
    print('***************** CV folds ****************')
    for i, cv_fold in enumerate(CVFold.query.all()):
        print(i, cv_fold)


def print_submission_similaritys():
    print('******** Submission similarities **********')
    for i, submission_similarity in enumerate(
            SubmissionSimilarity.query.all()):
        print(i, submission_similarity)


def create_user(name, password, lastname, firstname, email,
                access_level='user', hidden_notes='', linkedin_url='',
                twitter_url='', facebook_url='', google_url='', github_url='',
                website_url='', bio='', is_want_news=True):
    hashed_password = get_hashed_password(password)
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=email,
                access_level=access_level, hidden_notes=hidden_notes,
                linkedin_url=linkedin_url, twitter_url=twitter_url,
                facebook_url=facebook_url, google_url=google_url,
                github_url=github_url, website_url=website_url, bio=bio,
                is_want_news=is_want_news)

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
    user.is_authenticated = True


def get_sandbox(event, user):
    event_team = get_active_user_event_team(event, user)

    submission = Submission.query.filter_by(
        event_team=event_team, is_not_sandbox=False).one_or_none()
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


def ask_sign_up_team(event_name, team_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(
        event=event, team=team).one_or_none()
    if event_team is None:
        event_team = EventTeam(event=event, team=team)
        db.session.add(event_team)
        db.session.commit()


def sign_up_team(event_name, team_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(
        event=event, team=team).one_or_none()
    if event_team is None:
        ask_sign_up_team(event_name, team_name)
        event_team = EventTeam.query.filter_by(
            event=event, team=team).one_or_none()
    # submitting the starting kit for team
    from_submission_path = os.path.join(
        config.problems_path, event.problem.name, config.sandbox_d_name)
    make_submission_and_copy_files(
        event_name, team_name, config.sandbox_d_name, from_submission_path)
    for user in get_team_members(team):
        send_mail(user.email, 'signed up for {} as team {}'.format(
            event_name, team_name), '')
    event_team.approved = True
    db.session.commit()


def send_mail(to, subject, body):
    try:
        gmail_user = config.MAIL_USERNAME
        gmail_pwd = config.MAIL_PASSWORD
        smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
        smtpserver.ehlo()
        smtpserver.starttls()
        smtpserver.ehlo
        smtpserver.login(gmail_user, gmail_pwd)
        header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
            to, gmail_user, subject)
        smtpserver.sendmail(gmail_user, to, header + body)
    except Exception as e:
        logger.error('Mailing error: {}'.format(e))


def approve_user(user_name):
    user = User.query.filter_by(name=user_name).one()
    if user.access_level == 'asked':
        user.access_level = 'user'
    user.is_authenticated = True
    db.session.commit()
    send_mail(user.email, 'RAMP sign-up approved', '')


def make_event_admin(event_name, admin_name):
    event = Event.query.filter_by(name=event_name).one()
    admin = User.query.filter_by(name=admin_name).one()
    event_admin = EventAdmin.query.filter_by(
        event=event, admin=admin).one_or_none()
    if event_admin is None:
        event_admin = EventAdmin(event=event, admin=admin)
        db.session.commit()


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
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one()
    shutil.rmtree(submission.path)

    cv_folds = CVFold.query.filter_by(event=event).all()
    for cv_fold in cv_folds:
        submission_on_cv_fold = SubmissionOnCVFold.query.filter_by(
            submission=submission, cv_fold=cv_fold).one()
        cv_fold.submissions.remove(submission_on_cv_fold)

    delete_submission_similarity([submission])
    db.session.delete(submission)
    db.session.commit()
    compute_contributivity(event_name)
    compute_historical_contributivity(event_name)
    update_user_leaderboards(event_name, team_name)
    set_n_submissions(event_name)


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
        if last_submission is not None and last_submission.is_not_sandbox:
            now = datetime.datetime.utcnow()
            last = last_submission.submission_timestamp
            min_diff = event.min_duration_between_submissions
            diff = (now - last).total_seconds()
            if team.admin.access_level != 'admin' and diff < min_diff:
                raise TooEarlySubmissionError(
                    'You need to wait {} more seconds until next submission'
                    .format(int(min_diff - diff)))
        submission = Submission(name=submission_name, event_team=event_team)
        make_submission_on_cv_folds(event.cv_folds, submission)
        db.session.add(submission)
    else:
        # We allow resubmit for new or failing submissions
        if submission.is_not_sandbox and\
                (submission.state == 'new' or submission.is_error):
            submission.set_state('new')
            submission.submission_timestamp = datetime.datetime.utcnow()
            for submission_on_cv_fold in submission.on_cv_folds:
                submission_on_cv_fold.reset()
        else:
            raise DuplicateSubmissionError(
                'Submission "{}" of team "{}" at event "{}" exists already'.format(
                    submission_name, team_name, event_name))

    # All file names with at least a . in them
    deposited_f_name_list = os.listdir(submission_path)
    deposited_f_name_list = [
        f_name for f_name in deposited_f_name_list
        if f_name.find('.') >= 0]
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
            db.session.add(submission_file)

    # for remembering it in the sandbox view
    event_team.last_submission_name = submission_name
    db.session.commit()

    update_leaderboards(event_name)
    update_user_leaderboards(event_name, team.name)

    # We should copy files here
    return submission


def make_submission_and_copy_files(event_name, team_name, new_submission_name,
                                   from_submission_path):
    """Make submission and copy files to submission.path.

    Called from sign_up_team(), merge_teams(), fetch.add_models(),
    view.sandbox().
    """
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

    logger.info('Adding {}'.format(submission))
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
            subject = '{} was trained at {}'.format(
                submission, date_time_format(submission.training_timestamp))
            header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
                recipient_list, gmail_user, subject)
            body = ''
            for score in submission.scores:
                body += '{} = {}\n'.format(
                    score.score_name,
                    round(score.valid_score_cv_bag, score.precision))
            for recipient in recipient_list:
                smtpserver.sendmail(gmail_user, recipient, header + body)
        except Exception as e:
            logger.error('Mailing error: {}'.format(e))


def send_submission_mails(user, submission, event_team):
    #  later can be joined to the ramp admins
    event = event_team.event
    team = event_team.team
    recipient_list = config.ADMIN_MAILS[:]
    event_admins = EventAdmin.query.filter_by(event=event)
    recipient_list += [event_admin.admin.email for event_admin in event_admins]
    gmail_user = config.MAIL_USERNAME
    gmail_pwd = config.MAIL_PASSWORD
    smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    subject = 'fab train_test:e="{}",t="{}",s="{}"'.format(
        event.name, team.name.encode('utf-8'), submission.name.encode('utf-8'))
    header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
        recipient_list, gmail_user, subject.encode('utf-8'))
    # body = 'user = {}\nevent = {}\nserver = {}\nsubmission dir = {}\n'.format(
    #     user,
    #     event.name
    #     config.config_object.get_deployment_target(mode='train'),
    #     submission.path)
    body = 'user = {}\nevent = {}\nsubmission dir = {}\n'.format(
        user,
        event.name,
        submission.path)
    for recipient in recipient_list:
        smtpserver.sendmail(gmail_user, recipient, header + body)


def send_sign_up_request_mail(event, user):
    team = Team.query.filter_by(name=user.name).one()
    recipient_list = config.ADMIN_MAILS[:]
    event_admins = EventAdmin.query.filter_by(event=event)
    recipient_list += [event_admin.admin.email for event_admin in event_admins]
    gmail_user = config.MAIL_USERNAME
    gmail_pwd = config.MAIL_PASSWORD
    smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    subject = 'fab sign_up_team:e="{}",t="{}"'.format(
        event.name, team.name)
    header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
        recipient_list, gmail_user, subject)
    body = 'event = {}\n'.format(event.name)
    body += 'user = {}\n'.format(user.name.encode('utf-8'))
    body += 'name = {} {}\n'.format(
        user.firstname.encode('utf-8'), user.lastname.encode('utf-8'))
    body += 'email = {}\n'.format(user.email)
    body += 'linkedin = {}\n'.format(user.linkedin_url)
    body += 'twitter = {}\n'.format(user.twitter_url)
    body += 'facebook = {}\n'.format(user.facebook_url)
    body += 'github = {}\n'.format(user.github_url)
    body += 'notes = {}\n'.format(user.hidden_notes.encode('utf-8'))
    body += 'bio = {}\n\n'.format(user.bio.encode('utf-8'))
    url_approve = 'http://{}/events/{}/sign_up/{}'.format(
        config.current_server_name, event.name,
        user.name.encode('utf-8'))
    body += 'Click on this link to approve this user for this event: {}\n'.\
        format(url_approve)

    for recipient in recipient_list:
        smtpserver.sendmail(gmail_user, recipient, header + body)


def send_register_request_mail(user):
    recipient_list = config.ADMIN_MAILS[:]
    gmail_user = config.MAIL_USERNAME
    gmail_pwd = config.MAIL_PASSWORD
    smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)
    subject = 'fab approve_user:u="{}"'.format(user.name)
    header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
        recipient_list, gmail_user, subject)
    body = 'user = {}\n'.format(user.name.encode('utf-8'))
    body += 'name = {} {}\n'.format(
        user.firstname.encode('utf-8'), user.lastname.encode('utf-8'))
    body += 'email = {}\n'.format(user.email)
    body += 'linkedin = {}\n'.format(user.linkedin_url)
    body += 'twitter = {}\n'.format(user.twitter_url)
    body += 'facebook = {}\n'.format(user.facebook_url)
    body += 'github = {}\n'.format(user.github_url)
    body += 'notes = {}\n'.format(user.hidden_notes.encode('utf-8'))
    body += 'bio = {}\n\n'.format(user.bio.encode('utf-8'))
    url_approve = 'http://{}/sign_up/{}'.\
        format(config.current_server_name, user.name.encode('utf-8'))
    body += 'Click on the link to approve the registration of this user: {}\n'.\
        format(url_approve)

    for recipient in recipient_list:
        smtpserver.sendmail(gmail_user, recipient, header + body)




def get_earliest_new_submission(event_name=None):
    if event_name is None:
        new_submissions = Submission.query.filter_by(
            state='new').filter(Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
    # a fast fix: prefixing event name with 'not_' will exclude the event
    elif event_name[:4] == 'not_':
        event_name = event_name[4:]
        new_submissions = db.session.query(
            Submission, Event, EventTeam).filter(
            Event.name != event_name).filter(
            Event.id == EventTeam.event_id).filter(
            EventTeam.id == Submission.event_team_id).filter(
            Submission.state == 'new').filter(
            Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
        if new_submissions:
            new_submissions = list(zip(*new_submissions)[0])
        else:
            new_submissions = []
    else:
        new_submissions = db.session.query(
            Submission, Event, EventTeam).filter(
            Event.name == event_name).filter(
            Event.id == EventTeam.event_id).filter(
            EventTeam.id == Submission.event_team_id).filter(
            Submission.state == 'new').filter(
            Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
        if new_submissions:
            new_submissions = list(zip(*new_submissions)[0])
        else:
            new_submissions = []
    # Give ten seconds to upload submission files. Can be eliminated
    # once submission files go into database.
    new_submissions = [
        s for s in new_submissions
        if datetime.datetime.utcnow() - s.submission_timestamp >
        datetime.timedelta(0, 10)]

    if len(new_submissions) == 0:
        return None
    else:
        return new_submissions[0]


def set_n_submissions(event_name=None):
    if event_name is None:
        events = Event.query.all()
        for event in events:
            event.set_n_submissions()
    else:
        event = Event.query.filter_by(name=event_name).one()
        event.set_n_submissions()


def backend_train_test_loop(event_name=None, timeout=20,
                            is_compute_contributivity=True):
    event_names = set()
    while(True):
        earliest_new_submission = get_earliest_new_submission(event_name)
        logger.info('Automatic training {} at {}'.format(
            earliest_new_submission, datetime.datetime.utcnow()))
        if earliest_new_submission is not None:
            train_test_submission(earliest_new_submission)
            event_names.add(earliest_new_submission.event.name)
            update_leaderboards(earliest_new_submission.event.name)
            update_all_user_leaderboards(earliest_new_submission.event.name)
        else:
            # We only compute contributivity if nobody is waiting
            if is_compute_contributivity:
                for event_name in event_names:
                    compute_contributivity(event_name)
                    compute_historical_contributivity(event_name)
                    set_n_submissions(event_name)
            event_names = set()
        time.sleep(timeout)


def train_test_submissions(submissions=None, force_retrain_test=False):
    """Train and test submission.

    If submissions is None, trains and tests all submissions.
    """
    if submissions is None:
        submissions = Submission.query.filter(
            Submission.is_not_sandbox).order_by(Submission.id).all()
    for submission in submissions:
        train_test_submission(submission, force_retrain_test)


# For parallel call
def train_test_submission(submission, force_retrain_test=False):
    """We do it here so it's dockerizable."""

    submission.state = 'training'
    db.session.commit()

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
            submission.event.n_jobs))
        detached_submission_on_cv_folds = Parallel(
            n_jobs=submission.event.n_jobs, verbose=5)(
            delayed(train_test_submission_on_cv_fold)(
                submission_on_cv_fold, X_train, y_train, X_test, y_test,
                force_retrain_test)
            for submission_on_cv_fold in detached_submission_on_cv_folds)
        for detached_submission_on_cv_fold, submission_on_cv_fold in\
                zip(detached_submission_on_cv_folds, submission.on_cv_folds):
            try:
                submission_on_cv_fold.update(detached_submission_on_cv_fold)
            except Exception as e:
                detached_submission_on_cv_fold.state = 'training_error'
                log_msg, detached_submission_on_cv_fold.error_msg =\
                    _make_error_message(e)
                logger.error(
                    'Training {} failed with exception: \n{}'.format(
                        detached_submission_on_cv_fold, log_msg))
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
    # Means and stds were constructed on demand by fetching fold times.
    # It was slow because submission_on_folds contain also possibly large
    # predictions. If postgres solves this issue (which can be tested on
    # the mean and std scores on the private leaderbord), the
    # corresponding columns (which are now redundant) can be deleted in
    # Submission and this computation can also be deleted.
    submission.train_time_cv_mean = np.mean(
        [ts.train_time for ts in submission.on_cv_folds])
    submission.valid_time_cv_mean = np.mean(
        [ts.valid_time for ts in submission.on_cv_folds])
    submission.test_time_cv_mean = np.mean(
        [ts.test_time for ts in submission.on_cv_folds])
    submission.train_time_cv_std = np.std(
        [ts.train_time for ts in submission.on_cv_folds])
    submission.valid_time_cv_std = np.std(
        [ts.valid_time for ts in submission.on_cv_folds])
    submission.test_time_cv_std = np.std(
        [ts.test_time for ts in submission.on_cv_folds])
    db.session.commit()
    for score in submission.scores:
        logger.info('valid_score {} = {}'.format(
            score.score_name, score.valid_score_cv_bag))
        logger.info('test_score {} = {}'.format(
            score.score_name, score.test_score_cv_bag))

    send_trained_mails(submission)


def _make_error_message(e):
    """Make an error message in train/test.

    log_msg is the full error what we print into logger.error. error_msg
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
    # to avoid pickling error when joblib tries to return the model
    detached_submission_on_cv_fold.trained_submission = None
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
            detached_submission_on_cv_fold.workflow.train_submission(
                detached_submission_on_cv_fold.path, X, y, train_is)
        detached_submission_on_cv_fold.state = 'trained'
    except Exception as e:
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
        # Computing predictions on full training set
        y_pred = detached_submission_on_cv_fold.workflow.test_submission(
            detached_submission_on_cv_fold.trained_submission, X)
        assert_all_finite(y_pred)
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
    except Exception as e:
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
        y_pred = detached_submission_on_cv_fold.workflow.test_submission(
            detached_submission_on_cv_fold.trained_submission, X)
        assert_all_finite(y_pred)
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
    except Exception as e:
        detached_submission_on_cv_fold.state = 'testing_error'
        log_msg, detached_submission_on_cv_fold.error_msg = _make_error_message(e)
        logger.error(
            'Testing {} failed with exception: \n{}'.format(
                detached_submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    detached_submission_on_cv_fold.test_time = end - start


def compute_contributivity(event_name, start_time_stamp=None,
                           end_time_stamp=None, force_ensemble=False):
    compute_contributivity_no_commit(
        event_name, start_time_stamp, end_time_stamp, force_ensemble)
    db.session.commit()


# contributivity_df = score_plot_df.copy()
# contributivity_df = contributivity_df[
#     (contributivity_df['test pareto'] == 1) | (contributivity_df['valid pareto'] == 1)]
# contributivity_df['valid combined combined score'] = 0.0
# contributivity_df['test combined combined score'] = 0.0
# contributivity_df['valid combined foldwise score'] = 0.0
# contributivity_df['test combined foldwise score'] = 0.0
# for i, row in contributivity_df.iterrows():
#     print i
#     db_tools.compute_contributivity_no_commit(
#         event_name, end_time_stamp=row['submission timestamp (UTC)'])
#     contributivity_df.set_value(i, 'valid combined combined score', event.combined_combined_valid_score)
#     contributivity_df.set_value(i, 'test combined combined score', event.combined_combined_test_score)
#     contributivity_df.set_value(i, 'valid combined foldwise score', event.combined_foldwise_valid_score)
#     contributivity_df.set_value(i, 'test combined foldwise score', event.combined_foldwise_test_score)

def compute_contributivity_no_commit(
        event_name, start_time_stamp=None, end_time_stamp=None,
        force_ensemble=False):
    """Compute contributivity leaderboard scores.

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
    submissions = get_submissions(event_name=event_name)

    ground_truths_train = event.problem.ground_truths_train()
    ground_truths_test = event.problem.ground_truths_test()

    combined_predictions_list = []
    best_predictions_list = []
    combined_test_predictions_list = []
    best_test_predictions_list = []
    test_is_list = []
    for cv_fold in CVFold.query.filter_by(event=event).all():
        logger.info('{}'.format(cv_fold))
        ground_truths_valid = event.problem.ground_truths_valid(
            cv_fold.test_is)
        combined_predictions, best_predictions,\
            combined_test_predictions, best_test_predictions =\
            _compute_contributivity_on_fold(
                cv_fold, ground_truths_valid,
                start_time_stamp, end_time_stamp, force_ensemble)
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
    from model import _get_score_cv_bags
    # if there are no predictions to combine, it crashed
    combined_predictions_list = [c for c in combined_predictions_list
                                 if c is not None]
    if len(combined_predictions_list) > 0:
        combined_predictions, scores = _get_score_cv_bags(
            event, event.official_score_type, combined_predictions_list,
            ground_truths_train, test_is_list=test_is_list,
            is_return_combined_predictions=True)
        np.savetxt(
            'y_train_pred.csv', combined_predictions.y_pred, delimiter=',')
        logger.info('Combined combined valid score = {}'.format(scores))
        event.combined_combined_valid_score = float(scores[-1])
    else:
        event.combined_combined_valid_score = None

    best_predictions_list = [c for c in best_predictions_list
                             if c is not None]
    if len(best_predictions_list) > 0:
        scores = _get_score_cv_bags(
            event, event.official_score_type, best_predictions_list,
            ground_truths_train, test_is_list=test_is_list)
        logger.info('Combined foldwise best valid score = {}'.format(scores))
        event.combined_foldwise_valid_score = float(scores[-1])
    else:
        event.combined_foldwise_valid_score = None

    combined_test_predictions_list = [c for c in combined_test_predictions_list
                                      if c is not None]
    if len(combined_test_predictions_list) > 0:
        combined_predictions, scores = _get_score_cv_bags(
            event, event.official_score_type, combined_test_predictions_list,
            ground_truths_test, is_return_combined_predictions=True)
        np.savetxt(
            'y_test_pred.csv', combined_predictions.y_pred, delimiter=',')
        logger.info('Combined combined test score = {}'.format(scores))
        event.combined_combined_test_score = float(scores[-1])
    else:
        event.combined_combined_test_score = None

    best_test_predictions_list = [c for c in best_test_predictions_list
                                  if c is not None]
    if len(best_test_predictions_list) > 0:
        scores = _get_score_cv_bags(
            event, event.official_score_type, best_test_predictions_list,
            ground_truths_test)
        logger.info('Combined foldwise best valid score = {}'.format(scores))
        event.combined_foldwise_test_score = float(scores[-1])
    else:
        event.combined_foldwise_test_score = None

    return event.combined_combined_valid_score,\
        event.combined_foldwise_valid_score,\
        event.combined_combined_test_score,\
        event.combined_foldwise_test_score



def _compute_contributivity_on_fold(cv_fold, ground_truths_valid,
                                    start_time_stamp=None, end_time_stamp=None,
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
        and submission_on_fold.submission.is_not_sandbox
    ]
    # reset
    for submission_on_fold in selected_submissions_on_fold:
        submission_on_fold.best = False
        submission_on_fold.contributivity = 0.0
    # select submissions in time interval
    if start_time_stamp is not None:
        selected_submissions_on_fold = [
            submission_on_fold for submission_on_fold
            in selected_submissions_on_fold
            if submission_on_fold.submission.submission_timestamp >=
            start_time_stamp
        ]
    if end_time_stamp is not None:
        selected_submissions_on_fold = [
            submission_on_fold for submission_on_fold
            in selected_submissions_on_fold
            if submission_on_fold.submission.submission_timestamp <=
            end_time_stamp
        ]

    if len(selected_submissions_on_fold) == 0:
        return None, None, None, None
    # TODO: maybe this can be simplified. Don't need to get down
    # to prediction level.
    predictions_list = [
        submission_on_fold.valid_predictions
        for submission_on_fold in selected_submissions_on_fold]
    valid_scores = [
        submission_on_fold.official_score.valid_score
        for submission_on_fold in selected_submissions_on_fold]
    if cv_fold.event.official_score_type.is_lower_the_better:
        best_prediction_index = np.argmin(valid_scores)
    else:
        best_prediction_index = np.argmax(valid_scores)
    best_index_list = np.array([best_prediction_index])
    improvement = True
    while improvement and len(best_index_list) < cv_fold.event.max_n_ensemble:
        old_best_index_list = best_index_list
        best_index_list, score = get_next_best_single_fold(
            cv_fold.event, predictions_list, ground_truths_valid,
            best_index_list)
        improvement = len(best_index_list) != len(old_best_index_list)
        logger.info('\t{}: {}'.format(old_best_index_list, score))
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


def compute_historical_contributivity_no_commit(event_name):
    submissions = get_submissions(event_name=event_name)
    submissions.sort(key=lambda x: x.submission_timestamp, reverse=True)
    for submission in submissions:
        submission.historical_contributivity = 0.0
    for submission in submissions:
        submission.historical_contributivity += submission.contributivity
        submission_similaritys = SubmissionSimilarity.query.filter_by(
            type='target_credit', target_submission=submission).all()
        if submission_similaritys:
            # if a target team enters several credits to a source submission
            # we only take the latest
            submission_similaritys.sort(
                key=lambda x: x.timestamp, reverse=True)
            processed_submissions = []
            historical_contributivity = submission.historical_contributivity
            for submission_similarity in submission_similaritys:
                source_submission = submission_similarity.source_submission
                if source_submission not in processed_submissions:
                    partial_credit = historical_contributivity *\
                        submission_similarity.similarity
                    source_submission.historical_contributivity +=\
                        partial_credit
                    submission.historical_contributivity -= partial_credit
                    processed_submissions.append(source_submission)


def compute_historical_contributivity(event_name):
    compute_historical_contributivity_no_commit(event_name)
    db.session.commit()
    update_leaderboards(event_name)
    update_all_user_leaderboards(event_name)


def is_user_signed_up(event_name, user_name):
    for event_team in get_user_event_teams(event_name, user_name):
        if event_team.is_active and event_team.approved:
            return True
    return False


def is_user_asked_sign_up(event_name, user_name):
    for event_team in get_user_event_teams(event_name, user_name):
        if event_team.is_active and not event_team.approved:
            return True
    return False


def is_admin(event, user):
    if user.access_level == 'admin':
        return True
    event_admin = EventAdmin.query.filter_by(
        event=event, admin=user).one_or_none()
    if event_admin is None:
        return False
    else:
        return True


def is_public_event(event, user):
    if event is None:
        return False
    if user.access_level == 'asked':
        return False
    if event.is_public or is_admin(event, user):
        return True
    return False


def is_open_leaderboard(event, current_user):
    """
    True if current_user can look at submission of event.

    If submission is None, it is assumed to be the sandbox.
    """
    if not current_user.is_authenticated or not current_user.is_active:
        return False
    if is_admin(event, current_user):
        return True
    if not is_user_signed_up(event.name, current_user.name):
        return False
    if event.is_public_open:
        return True
    return False


def is_open_code(event, current_user, submission=None):
    """
    True if current_user can look at submission of event.

    If submission is None, it is assumed to be the sandbox.
    """
    if not current_user.is_authenticated or not current_user.is_active:
        return False
    if is_admin(event, current_user):
        return True
    if not is_user_signed_up(event.name, current_user.name):
        return False
    if submission is None:
        # It's probably stupid since we could just return True here, but this
        # access right thing will have to be cleaned up anyways
        submission = get_sandbox(event, current_user)
    if current_user in get_team_members(submission.event_team.team):
        return True
    if event.is_public_open:
        return True
    return False



def get_leaderboards(event_name, user_name=None):
    """
    Returns
    -------
    leaderboard_html_with_links : html string
    leaderboard_html_with_no_links : html string
    """

    submissions = get_submissions(event_name=event_name, user_name=user_name)
    submissions = [submission for submission in submissions
                   if submission.is_public_leaderboard and submission.is_valid]
    event = Event.query.filter_by(name=event_name).one()

    score_names = [score_type.name for score_type in event.score_types]
    scoress = np.array([
        [round(score.valid_score_cv_bag, score.precision)
         for score in submission.ordered_scores(score_names)]
        for submission in submissions
    ]).T
    leaderboard_df = pd.DataFrame()
    leaderboard_df['team'] = [
        submission.event_team.team.name for submission in submissions]
    leaderboard_df['submission'] = [
        submission.name_with_link for submission in submissions]
    leaderboard_df['submission no link'] = [
        submission.name[:20] for submission in submissions]
    leaderboard_df['contributivity'] = [
        int(round(100 * submission.contributivity))
        for submission in submissions]
    leaderboard_df['historical contributivity'] = [
        int(round(100 * submission.historical_contributivity))
        for submission in submissions]
    for score_name in score_names:  # to make sure the column is created
        leaderboard_df[score_name] = 0
    for score_name, scores in zip(score_names, scoress):
        leaderboard_df[score_name] = scores
    leaderboard_df['train time'] = [
        int(round(submission.train_time_cv_mean))
        for submission in submissions]
    leaderboard_df['test time'] = [
        int(round(submission.valid_time_cv_mean))
        for submission in submissions]
    leaderboard_df['submitted at (UTC)'] = [
        date_time_format(submission.submission_timestamp)
        for submission in submissions]
    sort_column = event.official_score_name
    leaderboard_df = leaderboard_df.sort_values(
        sort_column, ascending=event.official_score_type.is_lower_the_better)

    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html_with_links = leaderboard_df.drop(
        'submission no link', axis=1).to_html(**html_params)
    leaderboard_df = leaderboard_df.drop('submission', axis=1)
    leaderboard_df = leaderboard_df.rename(
        columns={'submission no link': 'submission'})
    leaderboard_html_no_links = leaderboard_df.to_html(**html_params)

    return (
        table_format(leaderboard_html_with_links),
        table_format(leaderboard_html_no_links)
    )


def get_private_leaderboards(event_name, user_name=None):
    """
    Returns
    -------
    leaderboard_html_with_links : html string
    leaderboard_html_with_no_links : html string
    """
    # start = time.time()

    submissions = get_submissions(event_name=event_name, user_name=user_name)
    submissions = [submission for submission in submissions
                   if submission.is_private_leaderboard]
    event = Event.query.filter_by(name=event_name).one()

    score_names = [score_type.name for score_type in event.score_types]
    scoresss = np.array([
        [[round(score.valid_score_cv_bag, score.precision),
          round(score.valid_score_cv_mean, score.precision),
          round(score.valid_score_cv_std, score.precision + 1),
          round(score.test_score_cv_bag, score.precision),
          round(score.test_score_cv_mean, score.precision),
          round(score.test_score_cv_std, score.precision + 1)
         ]
         for score in submission.ordered_scores(score_names)]
        for submission in submissions
    ])
    if len(submissions) > 0:
        scoresss = np.swapaxes(scoresss, 0, 1)
    leaderboard_df = pd.DataFrame()
    leaderboard_df['team'] = [
        submission.event_team.team.name for submission in submissions]
    leaderboard_df['submission'] = [
        submission.name_with_link for submission in submissions]
    for score_name in score_names:  # to make sure the column is created
        leaderboard_df[score_name + ' pub bag'] = 0
        leaderboard_df[score_name + ' pub mean'] = 0
        leaderboard_df[score_name + ' pub std'] = 0
        leaderboard_df[score_name + ' pr bag'] = 0
        leaderboard_df[score_name + ' pr mean'] = 0
        leaderboard_df[score_name + ' pr std'] = 0
    for score_name, scoress in zip(score_names, scoresss):
        leaderboard_df[score_name + ' pub bag'] = scoress[:, 0]
        leaderboard_df[score_name + ' pub mean'] = scoress[:, 1]
        leaderboard_df[score_name + ' pub std'] = scoress[:, 2]
        leaderboard_df[score_name + ' pr bag'] = scoress[:, 3]
        leaderboard_df[score_name + ' pr mean'] = scoress[:, 4]
        leaderboard_df[score_name + ' pr std'] = scoress[:, 5]
    leaderboard_df['contributivity'] = [
        int(round(100 * submission.contributivity))
        for submission in submissions]
    leaderboard_df['historical contributivity'] = [
        int(round(100 * submission.historical_contributivity))
        for submission in submissions]
    leaderboard_df['train time'] = [
        int(round(submission.train_time_cv_mean))
        for submission in submissions]
    leaderboard_df['trt std'] = [
        int(round(submission.train_time_cv_std))
        for submission in submissions]
    leaderboard_df['test time'] = [
        int(round(submission.valid_time_cv_mean))
        for submission in submissions]
    leaderboard_df['tet std'] = [
        int(round(submission.valid_time_cv_std))
        for submission in submissions]
    leaderboard_df['submitted at (UTC)'] = [
        date_time_format(submission.submission_timestamp)
        for submission in submissions]
    sort_column = event.official_score_name + ' pr bag'
    leaderboard_df = leaderboard_df.sort_values(
        sort_column, ascending=event.official_score_type.is_lower_the_better)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)

    # logger.info(u'private leaderboard construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    return table_format(leaderboard_html)


def update_leaderboards(event_name):
    private_leaderboard_html = get_private_leaderboards(
        event_name)
    leaderboards = get_leaderboards(event_name)
    failed_leaderboard_html = get_failed_leaderboard(event_name)
    new_leaderboard_html = get_new_leaderboard(event_name)

    event = Event.query.filter_by(name=event_name).one()
    event.private_leaderboard_html = private_leaderboard_html
    event.public_leaderboard_html_with_links = leaderboards[0]
    event.public_leaderboard_html_no_links = leaderboards[1]
    event.failed_leaderboard_html = failed_leaderboard_html
    event.new_leaderboard_html = new_leaderboard_html

    db.session.commit()


def update_user_leaderboards(event_name, user_name):
    logger.info('Leaderboard is updated for user {} in event {}.'.format(
        user_name, event_name))
    leaderboards = get_leaderboards(event_name, user_name)
    failed_leaderboard_html = get_failed_leaderboard(event_name, user_name)
    new_leaderboard_html = get_new_leaderboard(event_name, user_name)
    for event_team in get_user_event_teams(event_name, user_name):
        event_team.leaderboard_html = leaderboards[0]
        event_team.failed_leaderboard_html = failed_leaderboard_html
        event_team.new_leaderboard_html = new_leaderboard_html
    db.session.commit()


def update_all_user_leaderboards(event_name):
    event = Event.query.filter_by(name=event_name).one()
    event_teams = EventTeam.query.filter_by(event=event).all()
    for event_team in event_teams:
        user_name = event_team.team.name
        leaderboards = get_leaderboards(event_name, user_name)
        failed_leaderboard_html = get_failed_leaderboard(
            event_name, user_name)
        new_leaderboard_html = get_new_leaderboard(event_name, user_name)
        event_team.leaderboard_html = leaderboards[0]
        event_team.failed_leaderboard_html = failed_leaderboard_html
        event_team.new_leaderboard_html = new_leaderboard_html
    db.session.commit()


def get_failed_leaderboard(event_name, team_name=None, user_name=None):
    """
    Returns
    -------
    leaderboard_html : html string
    """

    # start = time.time()

    submissions = get_submissions(
        event_name=event_name, team_name=team_name, user_name=user_name)
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
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)

    # logger.info(u'failed leaderboard construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    return table_format(leaderboard_html)


def get_new_leaderboard(event_name, team_name=None, user_name=None):
    """
    Returns
    -------
    leaderboard_html : html string
    """
    # start = time.time()

    submissions = get_submissions(
        event_name=event_name, team_name=team_name, user_name=user_name)
    submissions = [submission for submission in submissions
                   if submission.state in ['new', 'training']
                   and submission.is_not_sandbox]

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
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )

    # logger.info(u'new leaderboard construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    leaderboard_html = leaderboard_df.to_html(**html_params)
    return table_format(leaderboard_html)


def get_submissions(event_name=None, team_name=None, user_name=None,
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
                if submissions_:
                    submissions = list(zip(*submissions_)[0])
                else:
                    submissions = []
            else:
                # All submissions for a given event by all the teams of a user
                submissions = []
                for event_team in get_user_event_teams(event_name, user_name):
                    submissions += db.session.query(
                        Submission).filter(
                        Event.name == event_name).filter(
                        Event.id == event_team.event_id).filter(
                        event_team.id == Submission.event_team_id).all()
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
            if submissions_:
                submissions = list(zip(*submissions_)[0])
            else:
                submissions = []
    return submissions


def get_submissions_of_state(state):
    return Submission.query.filter(Submission.state == state).all()


def print_submissions(event_name=None, team_name=None, submission_name=None):
    submissions = get_submissions(
        event_name=event_name, team_name=team_name,
        submission_name=submission_name)
    print('***************** List of submissions ****************')
    for submission in submissions:
        print(submission)
        print('\tstate = {}'.format(submission.state))
        print('\tcontributivity = {0:.2f}'.format(
            submission.contributivity))
        print('\thistorical contributivity = {0:.2f}'.format(
            submission.historical_contributivity))
        for score in submission.scores:
            print('\tscore_name = {}'.format(score.score_name))
            print('\t\tvalid_score_cv_mean = {0:.2f}'.format(
                score.valid_score_cv_mean))
            print('\t\tvalid_score_cv_bag = {0:.2f}'.format(
                float(score.valid_score_cv_bag)))
            print('\t\tvalid_score_cv_bags = {}'.format(
                score.valid_score_cv_bags))
            print('\t\ttest_score_cv_mean = {0:.2f}'.format(
                score.test_score_cv_mean))
            print('\t\ttest_score_cv_bag = {0:.2f}'.format(
                float(score.test_score_cv_bag)))
            print('\t\ttest_score_cv_bags = {}'.format(
                score.test_score_cv_bags))
        print('\tpath = {}'.format(submission.path))
        print('\tcv folds')
        submission_on_cv_folds = db.session.query(SubmissionOnCVFold).filter(
            SubmissionOnCVFold.submission == submission).all()
        for submission_on_cv_fold in submission_on_cv_folds:
            print('\t\t' + str(submission_on_cv_fold))


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
    best_valid_score = config.config_object.specific.score.worst
    best_test_score = config.config_object.specific.score.worst
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
    # start = time.time()
    user_interaction = UserInteraction(**kwargs)
    # logger.info(u'user interaction construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))
    # start = time.time()
    db.session.add(user_interaction)
    db.session.commit()
    # logger.info(u'user interaction db insertion takes {}ms'.format(
    #     int(1000 * (time.time() - start))))
    # with codecs.open(config.user_interactions_f_name, 'a+') as f:
    #     f.write(user_interaction.__repr__() + '\n')

# The following function was implemented to handle user interaction dump
# but it turned out that the db insertion was not the CPU sink. Keep it
# for a while if the site is still slow.
# def update_user_interactions():
#     f = codecs.open(config.user_interactions_f_name, 'r')
#     shutil.move(
#         config.user_interactions_f_name,
#         config.user_interactions_f_name + '.bak')
#     for line in f:
#         user_interaction = UserInteraction(line)
#         db.session.add(user_interaction)
#     db.session.commit()


def print_user_interactions():
    print('*********** User interactions ****************')
    for user_interaction in UserInteraction.query.all():
        print(date_time_format(user_interaction.timestamp),
              user_interaction.user, user_interaction.interaction)
        if user_interaction.submission_file_diff is not None:
            print(user_interaction.submission_file_diff)
        if user_interaction.submission_file_similarity is not None:
            print(user_interaction.submission_file_similarity)
        if user_interaction.submission is not None:
            print(user_interaction.submission)


def get_user_interactions_df():
    """
    Returns
    -------
    user_interactions_df : pd.DataFrame
    """

    user_interactions = UserInteraction.query.all()

    columns = ['timestamp (UTC)',
               'IP',
               'interaction',
               'user',
               'event',
               'team',
               'submission_id',
               'submission',
               'file',
               'code similarity',
               'diff']

    def user_name(user_interaction):
        user = user_interaction.user
        if user is None:
            return ''
        else:
            return user.name

    def event_name(user_interaction):
        event_team = user_interaction.event_team
        if event_team is None:
            return ''
        else:
            return event_team.event.name

    def team_name(user_interaction):
        event_team = user_interaction.event_team
        if event_team is None:
            return ''
        else:
            return event_team.team.name

    def submission_id(user_interaction):
        submission = user_interaction.submission
        if submission is None:
            return -1
        else:
            return submission.id

    def submission_name(user_interaction):
        submission = user_interaction.submission
        if submission is None:
            return ''
        else:
            return submission.name_with_link

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
                      user_interaction.ip,
                      user_interaction.interaction,
                      user_name(user_interaction),
                      event_name(user_interaction),
                      team_name(user_interaction),
                      submission_id(user_interaction),
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
    return user_interactions_df


def get_user_interactions():
    """
    Returns
    -------
    user_interactions_html : html string
    """

    user_interactions_df = get_user_interactions_df()
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    user_interactions_html = user_interactions_df.to_html(**html_params)
    return table_format(user_interactions_html)


def get_source_submissions(submission):
    submissions = Submission.query.filter_by(
        event_team=submission.event_team).all()
    users = get_team_members(submission.team)
    for user in users:
        user_interactions = UserInteraction.query.filter_by(
            user=user, interaction='looking at submission').all()
        submissions += [user_interaction.submission for
                        user_interaction in user_interactions if
                        user_interaction.event == submission.event]
    submissions = list(set(submissions))
    submissions = [s for s in submissions if
                   s.submission_timestamp < submission.submission_timestamp]
    submissions.sort(key=lambda x: x.submission_timestamp, reverse=True)
    return submissions
