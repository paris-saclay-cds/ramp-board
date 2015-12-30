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
    WorkflowElement, Extension,\
    CVFold, SubmissionOnCVFold, DetachedSubmissionOnCVFold,\
    UserInteraction,\
    NameClashError, MergeTeamError, TooEarlySubmissionError,\
    DuplicateSubmissionError, MissingSubmissionFileError,\
    MissingExtensionError,\
    combine_predictions_list, get_next_best_single_fold,\
    get_active_user_team, get_user_teams, get_n_team_members,\
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
    (Using bcrypt, the salt is saved into the hash itself)"""
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hased password. Using bcrypt, the salt is saved into the
    hash itself"""
    return bcrypt.checkpw(plain_text_password, hashed_password)


def add_users_from_file(users_to_add_f_name):
    # For now just saves the passwords and returns the pandas dataframe.
    users_to_add = pd.read_csv(users_to_add_f_name)
    if users_to_add_f_name.split('.')[-1] != 'w_pwd':
        words = xp.locate_wordfile()
        mywords = xp.generate_wordlist(
            wordfile=words, min_length=4, max_length=6)
        users_to_add['password'] = [
            xp.generate_xkcdpassword(mywords, numwords=4)
            for name in users_to_add['name']]
        # temporarily while we don't implement pwd recovery
        users_to_add.to_csv(users_to_add_f_name + '.w_pwd')
    return users_to_add


def send_password_mails(users_to_add_f_name):
    #  later can be joined to the ramp admins
    gmail_user = config.MAIL_USERNAME
    gmail_pwd = config.MAIL_PASSWORD
    smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(gmail_user, gmail_pwd)

    users_to_add = pd.read_csv(users_to_add_f_name)
    subject = '{} RAMP information'.format(
        config.config_object.specific.ramp_title)
    for _, u in users_to_add.iterrows():
        logger.info('Sending mail to {}'.format(u['email']))
        header = 'To: {}\nFrom: {}\nSubject: {}\n'.format(
            u['email'], gmail_user, subject)
        body = 'Dear {},\n\n'.format(u['firstname'])
        body += 'Here is your login and other information for the {} RAMP:\n\n'.format(
            config.config_object.specific.ramp_title)
        body += 'username: {}\n'.format(u['name'])
        body += 'password: {}\n'.format(u['password'])
        body += 'submission site: http://{}:{}\n'.format(
            config.config_object.web_server, config.config_object.server_port)
        if config.opening_timestamp is not None:
            body += 'opening at (UTC) {}\n'.format(
                date_time_format(config.opening_timestamp))
        if config.public_opening_timestamp is not None:
            body += 'opening of the collaborative phase at (UTC) {}\n'.format(
                date_time_format(config.public_opening_timestamp))
        if config.closing_timestamp is not None:
            body += 'closing at (UTC) {}\n'.format(
                date_time_format(config.closing_timestamp))
        smtpserver.sendmail(gmail_user, u['email'], header + body)


def setup_workflow_element_types():
    extension_names = ['py', 'R', 'txt', 'csv']
    for name in extension_names:
        extension = Extension.query.filter_by(name=name).one_or_none()
        if extension is None:
            db.session.add(Extension(name=name))
    db.session.commit()

    submission_file_types = [
        ('code', True, 10 ** 5),
        ('text', True, 10 ** 5),
        ('data', False, 10 ** 8)
    ]
    for (name, is_editable, max_size) in submission_file_types:
        submission_file_type = SubmissionFileType.query.filter_by(
            name=name).one_or_none()
        if submission_file_type is None:
            db.session.add(SubmissionFileType(
                name=name, is_editable=is_editable, max_size=max_size))
    db.session.commit()

    submission_file_type_extensions = [
        ('code', 'py'),
        ('code', 'R'),
        ('text', 'txt'),
        ('data', 'csv')
    ]
    for (type_name, extension_name) in submission_file_type_extensions:
        submission_file_type = SubmissionFileType.query.filter_by(
            name=type_name).one()
        extension = Extension.query.filter_by(name=extension_name).one()
        type_extension = SubmissionFileTypeExtension.query.filter_by(
            type=submission_file_type, extension=extension).one_or_none()
        if type_extension is None:
            db.session.add(SubmissionFileTypeExtension(
                type=submission_file_type, extension=extension))
    db.session.commit()

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
    for name, type_name in workflow_element_types:
        submission_file_type = SubmissionFileType.query.filter_by(
            name=type_name).one()
        workflow_element_type = WorkflowElementType.query.filter_by(
            name=name, type=submission_file_type).one_or_none()
        if workflow_element_type is None:
            db.session.add(WorkflowElementType(
                name=name, type=submission_file_type))
    db.session.commit()


def setup_problem(workflow_elements):
    for workflow_element_dict in workflow_elements:
        name = workflow_element_dict['name']
        try:
            type = workflow_element_dict['type']
            workflow_element = WorkflowElement(
                name=type, name_in_workflow=name)
        except KeyError:
            workflow_element = WorkflowElement(name=name)
        db.session.add(workflow_element)
    db.session.commit()


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
    # submitting the starting kit for default team
    make_submission_and_copy_files(
        name, config.sandbox_d_name, config.sandbox_path)
    return user


def validate_user(user):
    # from 'asked' to 'user'
    user.access_level = 'user'


def get_sandbox(user):
    team = get_active_user_team(user)
    submission = Submission.query.filter(
        team.id == Submission.team_id).filter(
        Submission.name == config.sandbox_d_name).one()
    return submission


def print_users():
    print('***************** List of users ****************')
    for user in User.query.order_by(User.id):
        print('{} belongs to teams:'.format(user))
        for team in get_user_teams(user):
            print('\t{}'.format(team))
        print('Sandbox = {}'.format(get_sandbox(user)))


######################### Teams ###########################


def merge_teams(name, initiator_name, acceptor_name):
    initiator = Team.query.filter_by(name=initiator_name).one()
    acceptor = Team.query.filter_by(name=acceptor_name).one()
    if not initiator.is_active:
        raise MergeTeamError('Merge initiator is not active')
    if not acceptor.is_active:
        raise MergeTeamError('Merge acceptor is not active')
    # Meaning they are already in another team for the ramp

    # Testing if team size is <= max_members_per_team
    n_members_initiator = get_n_team_members(initiator)
    n_members_acceptor = get_n_team_members(acceptor)
    n_members_new = n_members_initiator + n_members_acceptor
    if n_members_new > config.max_members_per_team:
        raise MergeTeamError(
            'Too big team: new team would be of size {}, the max is {}'.format(
                n_members_new, config.max_members_per_team))

    members_initiator = get_team_members(initiator)
    members_acceptor = get_team_members(acceptor)

    # Testing if team (same members) exists under a different name. If the
    # name is the same, we break. If the loop goes through, we add new team.
    members_set = set(members_initiator).union(set(members_acceptor))
    for team in Team.query:
        if members_set == set(get_team_members(team)):
            if name == team.name:
                # ok, but don't add new team, just set them to inactive
                initiator.is_active = False
                acceptor.is_active = False
                return team 
            raise MergeTeamError(
                'Team exists with the same members, team name = {}'.format(
                    team.name))
    else:
        team = Team(name=name, admin=initiator.admin,
                    initiator=initiator, acceptor=acceptor)
        db.session.add(team)
    initiator.is_active = False
    acceptor.is_active = False
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        try:
            Team.query.filter_by(name=name).one()
            raise NameClashError('team name is already in use')
        except NoResultFound:
            raise e
    logger.info('Merging {} and {} into {}'.format(initiator, acceptor, team))
    # submitting the starting kit for merged team
    make_submission_and_copy_files(
        name, config.sandbox_d_name, config.sandbox_path)
    return team


def print_active_teams():
    print('***************** List of active teams ****************')
    for team in Team.query.filter(Team.is_active):
        print('{} members:'.format(team))
        for member in get_team_members(team):
            print('\t{}'.format(member))


######################### Submissions ###########################

def set_state(team_name, submission_name, state):
    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        name=submission_name, team=team).one()
    submission.set_state(state)
    db.session.commit()


def delete_submission(team_name, submission_name):
    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        name=submission_name, team=team).one()
    shutil.rmtree(submission.path)
    db.session.delete(submission)
    db.session.commit()


def make_submission_on_cv_folds(cv_folds, submission):
    for cv_fold in cv_folds:
        submission_on_cv_fold = SubmissionOnCVFold(
            submission=submission, cv_fold=cv_fold)
        db.session.add(submission_on_cv_fold)


def make_submission(team_name, submission_name, submission_path):
    # TODO: to call unit tests on submitted files. Those that are found
    # in the table that describes the workflow. For the rest just check
    # maybe size
    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        name=submission_name, team=team).one_or_none()
    cv_folds = CVFold.query.all()
    workflow_elements = WorkflowElement.query.all()
    if submission is None:
        # Checking if submission is too early
        all_submissions = Submission.query.filter_by(team=team).order_by(
            Submission.submission_timestamp).all()
        last_submission = None if len(all_submissions) == 0 \
            else all_submissions[-1]
        if last_submission is not None and\
                last_submission.name != config.sandbox_d_name:
            now = datetime.datetime.utcnow()
            last = last_submission.submission_timestamp
            min_diff = config.min_duration_between_submissions
            diff = (now - last).total_seconds()
            if diff < min_diff:
                raise TooEarlySubmissionError(
                    'You need to wait {} more seconds until next submission'
                    .format(int(min_diff - diff)))
        submission = Submission(name=submission_name, team=team)
        db.session.add(submission)
    else:
        # We allow resubmit for new or failing submissions
        if submission.name != config.sandbox_d_name and\
                (submission.state == 'new' or submission.is_error):
            submission.state = 'new'
            submission.submission_timestamp = datetime.datetime.utcnow()
            for submission_on_cv_fold in submission.on_cv_folds:
                # couldn't figure out how to reset to default values
                db.session.delete(submission_on_cv_fold)
        else:
            raise DuplicateSubmissionError(
                'Submission "{}" of team "{}" exists already'.format(
                    submission_name, team_name))

    deposited_f_name_list = os.listdir(submission_path)
    # TODO: more error checking
    deposited_types = [f_name.split('.')[0]
                       for f_name in deposited_f_name_list]
    deposited_extensions = [f_name.split('.')[1]
                            for f_name in deposited_f_name_list]
    for workflow_element in workflow_elements:
        # We find all files with matching names to workflow_element.name.
        # If none found, raise error.
        # Then look for one that has a legal extension. If none found, 
        # raise error. If there are several ones, for now we use the first 
        # matching file.

        name = workflow_element.name
        i_names = [i for i in range(len(deposited_types)) 
                   if deposited_types[i] == name]
        if len(i_names) == 0:
            db.session.rollback()
            raise MissingSubmissionFileError('{}/{}/{}: {}'.format(
                team_name, submission_name, name, submission_path))

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
            raise MissingExtensionError('{}/{}/{}/{}: {}'.format(
                team_name, submission_name, name, extensions, submission_path))

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
    make_submission_on_cv_folds(cv_folds, submission)
    # for remembering it in the sandbox view
    team.last_submission_name = submission_name
    db.session.commit()

    # We should copy files here
    return submission


def make_submission_and_copy_files(team_name, new_submission_name,
                                   from_submission_path):
    """Called from create_user(), merge_teams(), fetch.add_models(),
    view.sandbox()."""

    submission = make_submission(
        team_name, new_submission_name, from_submission_path)
    team_path, new_submission_path = submission.get_paths(
        config.submissions_path)

    if not os.path.exists(team_path):
        os.mkdir(team_path)
    open(os.path.join(team_path, '__init__.py'), 'a').close()
    # clean up the model directory in case it's a resubmission
    if os.path.exists(new_submission_path):
        shutil.rmtree(new_submission_path)
    os.mkdir(new_submission_path)
    open(os.path.join(new_submission_path, '__init__.py'), 'a').close()

    # copy the submission files into the model directory, should all this
    # probably go to Submission
    for f_name in submission.f_names:
        src = os.path.join(from_submission_path, f_name)
        dst = os.path.join(new_submission_path, f_name)
        shutil.copy2(src, dst)  # copying also metadata
        logger.info('Copying {} to {}'.format(src, dst))

    logger.info("Adding submission={}".format(submission))
    return submission


def train_test_submissions(submissions=None, force_retrain_test=False):
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

    specific = config.config_object.specific
    X_train, y_train = specific.get_train_data()
    X_test, y_test = specific.get_test_data()

    # Parallel, dict
    if config.is_parallelize:
        # We are using 'threading' so train_test_submission_on_cv_fold
        # updates the detached submission_on_cv_fold objects. If it doesn't
        # work, we can go back to multiprocessing and
        print config.config_object.n_cpus
        detached_submission_on_cv_folds = Parallel(
            n_jobs=config.config_object.n_cpus, verbose=5)(
            delayed(train_test_submission_on_cv_fold)(
                submission_on_cv_fold, X_train, y_train, X_test, y_test,
                force_retrain_test)
            for submission_on_cv_fold in detached_submission_on_cv_folds)
    else:
        # detached_submission_on_cv_folds = []
        for submission_on_cv_fold in detached_submission_on_cv_folds:
            train_test_submission_on_cv_fold(
                submission_on_cv_fold, X_train, y_train, X_test, y_test,
                force_retrain_test)
    for detached_submission_on_cv_fold, submission_on_cv_fold in\
            zip(detached_submission_on_cv_folds, submission.on_cv_folds):
        submission_on_cv_fold.update(detached_submission_on_cv_fold)
    submission.training_timestamp = datetime.datetime.utcnow()
    submission.set_state_after_training()
    submission.compute_test_score_cv_bag()
    submission.compute_valid_score_cv_bag()
    db.session.commit()
    logger.info('valid_score = {}'.format(submission.valid_score_cv_bag))
    logger.info('test_score = {}'.format(submission.test_score_cv_bag))


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


def train_test_submission_on_cv_fold(submission_on_cv_fold, X_train, y_train,
                                     X_test, y_test, force_retrain_test=False):
    train_submission_on_cv_fold(submission_on_cv_fold, X_train, y_train,
                                force_retrain=force_retrain_test)
    if 'error' not in submission_on_cv_fold.state:
        test_submission_on_cv_fold(submission_on_cv_fold, X_test, y_test,
                                   force_retest=force_retrain_test)
    # When called in a single thread, we don't need the return value,
    # submission_on_cv_fold is modified in place. When called in parallel
    # multiprocessing mode, however, copies are made when the function is
    # called, so we have to explicitly return the modified object (so it is
    # ercopied into the original object)
    return submission_on_cv_fold


def train_submission_on_cv_fold(submission_on_cv_fold, X, y,
                                force_retrain=False):
    if submission_on_cv_fold.state not in ['new', 'checked']\
            and not force_retrain:
        if 'error' in submission_on_cv_fold.state:
            logger.error('Trying to train failed {}'.format(
                submission_on_cv_fold))
        else:
            logger.info('Already trained {}'.format(submission_on_cv_fold))
        return

    # so to make it importable, TODO: should go to make_submission
    # open(os.path.join(self.submission.path, '__init__.py'), 'a').close()

    train_is = submission_on_cv_fold.train_is
    specific = config.config_object.specific

    logger.info('Training {}'.format(submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        submission_on_cv_fold.trained_submission = specific.train_submission(
            submission_on_cv_fold.module, X, y, train_is)
        submission_on_cv_fold.state = 'trained'
    except Exception, e:
        submission_on_cv_fold.state = 'training_error'
        log_msg, submission_on_cv_fold.error_msg = _make_error_message(e)
        logger.error(
            'Training {} failed with exception: \n{}'.format(
                submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    submission_on_cv_fold.train_time = end - start

    logger.info('Validating {}'.format(submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        predictions = specific.test_submission(
            submission_on_cv_fold.trained_submission, X, range(len(y)))
        if predictions.n_samples == len(y):
            submission_on_cv_fold.full_train_predictions = predictions
            submission_on_cv_fold.state = 'validated'
        else:
            submission_on_cv_fold.error_msg = 'Wrong output dimension in ' +\
                'predict: {} instead of {}'.format(
                    predictions.n_samples, len(y))
            submission_on_cv_fold.state = 'validating_error'
            logger.error(
                'Validating {} failed with exception: \n{}'.format(
                    submission_on_cv_fold.error_msg))
            return
    except Exception, e:
        submission_on_cv_fold.state = 'validating_error'
        log_msg, submission_on_cv_fold.error_msg = _make_error_message(e)
        logger.error(
            'Validating {} failed with exception: \n{}'.format(
                submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    submission_on_cv_fold.validtime = end - start


def test_submission_on_cv_fold(submission_on_cv_fold, X, y,
                               force_retest=False):
    if submission_on_cv_fold.state not in\
            ['new', 'checked', 'trained', 'validated'] and not force_retest:
        if 'error' in submission_on_cv_fold.state:
            logger.error('Trying to test failed {}'.format(
                submission_on_cv_fold))
        else:
            logger.info('Already tested {}'.format(submission_on_cv_fold))
        return

    specific = config.config_object.specific

    logger.info('Testing {}'.format(submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        predictions = specific.test_submission(
            submission_on_cv_fold.trained_submission, X, range(len(y)))
        if predictions.n_samples == len(y):
            submission_on_cv_fold.test_predictions = predictions
            submission_on_cv_fold.state = 'tested'
        else:
            submission_on_cv_fold.error_msg = 'Wrong output dimension in ' +\
                'predict: {} instead of {}'.format(
                    predictions.n_samples, len(y))
            submission_on_cv_fold.state = 'testing_error'
            logger.error(
                'Testing {} failed with exception: \n{}'.format(
                    submission_on_cv_fold.error_msg))
    except Exception, e:
        submission_on_cv_fold.state = 'testing_error'
        log_msg, submission_on_cv_fold.error_msg = _make_error_message(e)
        logger.error(
            'Testing {} failed with exception: \n{}'.format(
                submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    submission_on_cv_fold.test_time = end - start


def compute_contributivity(force_ensemble=False):
    """Computes contributivity leaderboard scores.

    Parameters
    ----------
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

    true_predictions_train = generic.get_true_predictions_train()
    true_predictions_test = generic.get_true_predictions_test()

    combined_predictions_list = []
    best_predictions_list = []
    combined_test_predictions_list = []
    best_test_predictions_list = []
    test_is_list = []
    for cv_fold in CVFold.query.all():
        logger.info('{}'.format(cv_fold))
        combined_predictions, best_predictions,\
            combined_test_predictions, best_test_predictions =\
            compute_contributivity_on_fold(cv_fold, force_ensemble)
        # TODO: if we do asynchron CVs, this has to be revisited
        if combined_predictions is None:
            logger.info('No submissions to combine')
            return
        combined_predictions_list.append(combined_predictions)
        best_predictions_list.append(best_predictions)
        combined_test_predictions_list.append(combined_test_predictions)
        best_test_predictions_list.append(best_test_predictions)
        test_is_list.append(cv_fold.test_is)
    for submission in Submission.query.all():
        submission.set_contributivity()
    db.session.commit()
    from model import _get_score_cv_bags
    # if there are no predictions to combine, it crashed
    combined_predictions_list = [c for c in combined_predictions_list
                                 if c is not None]
    if len(combined_predictions_list) > 0:
        logger.info('Combined combined valid score = {}'.format(
            _get_score_cv_bags(combined_predictions_list,
                               true_predictions_train, test_is_list)))

    best_predictions_list = [c for c in best_predictions_list
                             if c is not None]
    if len(best_predictions_list) > 0:
        logger.info('Combined foldwise best valid score = {}'.format(
            _get_score_cv_bags(best_predictions_list,
                               true_predictions_train, test_is_list)))

    combined_test_predictions_list = [c for c in combined_test_predictions_list
                                      if c is not None]
    if len(combined_test_predictions_list) > 0:
        logger.info('Combined combined test score = {}'.format(
            _get_score_cv_bags(combined_test_predictions_list,
                               true_predictions_test)))

    best_test_predictions_list = [c for c in best_test_predictions_list
                                  if c is not None]
    if len(best_test_predictions_list) > 0:
        logger.info('Combined foldwise best test score = {}'.format(
            _get_score_cv_bags(best_test_predictions_list,
                               true_predictions_test)))


def compute_contributivity_on_fold(cv_fold, force_ensemble=False):
    """Constructs the best model combination on a single fold, using greedy
    forward selection with replacement. See
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
        print 'bla'
        return None, None, None, None
    true_predictions = generic.get_true_predictions_valid(cv_fold.test_is)
    # TODO: maybe this can be simplified. Don't need to get down
    # to prediction level.
    predictions_list = [
        submission_on_fold.valid_predictions
        for submission_on_fold in selected_submissions_on_fold]
    valid_scores = [
        submission_on_fold.valid_score
        for submission_on_fold in selected_submissions_on_fold]
    best_prediction_index = np.argmax(valid_scores)
    best_index_list = np.array([best_prediction_index])
    improvement = True
    while improvement and len(best_index_list) < config.max_n_ensemble:
        old_best_index_list = best_index_list
        best_index_list, score = get_next_best_single_fold(
            predictions_list, true_predictions, best_index_list)
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


def get_public_leaderboard(team_name=None, user=None, is_open_code=True):
    """
    Returns
    -------
    leaderboard_html : html string
    """

    # We can't query on non-hybrid properties like Submission.name_with_link,
    # so first we make the join then we extract the class members,
    # including @property members (that can't be compiled into
    # SQL queries)
    if team_name is not None:
        submissions_teams = db.session.query(Submission, Team).filter(
            Team.id == Submission.team_id).filter(
            Submission.is_public_leaderboard).filter(
            Submission.name != config.sandbox_d_name).filter(
            Team.name == team_name).all()
    elif user is not None:
        submissions_teams = []
        for team_name in [team.name for team in get_user_teams(user)]:
            submissions_teams += db.session.query(Submission, Team).filter(
                Team.id == Submission.team_id).filter(
                Team.name == team_name).filter(
                Submission.name != config.sandbox_d_name).filter(
                Submission.is_public_leaderboard).all()
    else:
        submissions_teams = db.session.query(Submission, Team).filter(
            Team.id == Submission.team_id).filter(
            Submission.is_public_leaderboard).filter(
            Submission.name != config.sandbox_d_name).all()

    columns = ['team',
               'submission',
               'score',
               'contributivity',
               'train time',
               'test time',
               'submitted at (UTC)']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [team.name,
                      submission.name_with_link if is_open_code
                      else submission.name[:20],
                      round(submission.valid_score_cv_bag, 3),
                      int(100 * submission.contributivity + 0.5),
                      int(submission.train_time_cv_mean + 0.5),
                      int(submission.valid_time_cv_mean + 0.5),
                      date_time_format(submission.submission_timestamp)])}
        for submission, team in submissions_teams
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    leaderboard_df = leaderboard_df.sort('contributivity', ascending=False)
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


def get_private_leaderboard():
    """
    Returns
    -------
    leaderboard_html : html string
    """

    submissions_teams = db.session.query(Submission, Team).filter(
        Team.id == Submission.team_id).filter(
        Submission.is_private_leaderboard).filter(
        Submission.name != config.sandbox_d_name).all()

    columns = ['team',
               'submission',
               'public score',
               'private score',
               'contributivity',
               'train time',
               'test time',
               'submitted at (UTC)']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [team.name,
                      submission.name_with_link,
                      round(submission.valid_score_cv_bag, 3),
                      round(submission.test_score_cv_bag, 3),
                      int(100 * submission.contributivity + 0.5),
                      int(submission.train_time_cv_mean + 0.5),
                      int(submission.valid_time_cv_mean + 0.5),
                      date_time_format(submission.submission_timestamp)])}
        for submission, team in submissions_teams
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    leaderboard_df = leaderboard_df.sort('contributivity', ascending=False)
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


def get_new_submissions(user=None):
    """
    Returns
    -------
    leaderboard_html : html string
    """
    if user is None:
        submissions_teams = db.session.query(Submission, Team).filter(
            Team.id == Submission.team_id).filter(
            Submission.state == 'new').filter(
            Submission.name != config.sandbox_d_name).order_by(
            Submission.submission_timestamp).all()
    else:
        submissions_teams = []
        for team_name in [team.name for team in get_user_teams(user)]:
            submissions_teams += db.session.query(Submission, Team).filter(
                Team.id == Submission.team_id).filter(
                Team.name == team_name).filter(
                Submission.state == 'new').filter(
                Submission.name != config.sandbox_d_name).order_by(
                Submission.submission_timestamp).all()
    columns = ['team',
               'submission',
               'submitted at (UTC)']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [team.name,
                      submission.name_with_link,
                      date_time_format(submission.submission_timestamp)])}
        for submission, team in submissions_teams
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


def get_team_submissions(team_name, submission_name=None):
    team = Team.query.filter(Team.name == team_name).one()
    if submission_name is None:
        submissions = Submission.query.filter(
            team.id == Submission.team_id).all()
    else:
        submissions = Submission.query.filter(
            team.id == Submission.team_id).filter(
            submission_name == Submission.name).all()
    return submissions


def get_submissions_of_state(state):
    return Submission.query.filter(Submission.state == state).all()


def get_earliest_new_submission():
    new_submissions = Submission.query.filter_by(
        Submission.state == 'new').order_by(
        Submission.submission_timestamp).all()
    if len(new_submissions) == 0:
        return None
    else:
        return new_submissions[0]


def get_failed_submissions(user=None):
    """
    Returns
    -------
    leaderboard_html : html string
    """
    if user is None:
        submissions_teams = db.session.query(Submission, Team).filter(
            Team.id == Submission.team_id).filter(
            Submission.is_error).order_by(
            Submission.submission_timestamp).all()
    else:
        submissions_teams = []
        for team_name in [team.name for team in get_user_teams(user)]:
            submissions_teams += db.session.query(Submission, Team).filter(
                Team.id == Submission.team_id).filter(
                Team.name == team_name).filter(
                Submission.is_error).order_by(
                Submission.submission_timestamp).all()
    columns = ['team',
               'submission',
               'submitted at (UTC)',
               'error']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [team.name,
                      submission.name_with_link,
                      date_time_format(submission.submission_timestamp),
                      submission.state_with_link])}
        for submission, team in submissions_teams
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


def print_submissions(submissions=None):
    if submissions is None:
        submissions = Submission.query.order_by(Submission.id).all()
    print('***************** List of submissions ****************')
    for submission in submissions:
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


def print_cv_folds():
    print('***************** CV folds ****************')
    for i, cv_fold in enumerate(CVFold.query.all()):
        print i, cv_fold


######################### CVFold ###########################

def reset_cv_folds():
    """Changing the CV scheme without deleting the submissions."""
    cv_folds = CVFold.query.all()
    for cv_fold in cv_folds:
        db.session.delete(cv_fold)
    db.session.commit()
    specific = config.config_object.specific
    _, y_train = specific.get_train_data()
    cv = specific.get_cv(y_train)
    add_cv_folds(cv)
    cv_folds = CVFold.query.all()
    submissions = Submission.query.all()
    for submission in submissions:
        make_submission_on_cv_folds(cv_folds, submission)
    db.session.commit()


def add_cv_folds(cv):
    for train_is, test_is in cv:
        cv_fold = CVFold(train_is=train_is, test_is=test_is)
        db.session.add(cv_fold)
    db.session.commit()


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
