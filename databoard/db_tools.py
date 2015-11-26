import bcrypt
import timeit
import logging
import datetime
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from databoard import db

from databoard.model import User, Team, Submission, SubmissionFile,\
    CVFold, SubmissionOnCVFold, DetachedSubmissionOnCVFold,\
    NameClashError, MergeTeamError,\
    DuplicateSubmissionError, max_members_per_team
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

import databoard.config as config


logger = logging.getLogger('databoard')
pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output


def _date_time_format(date_time):
    return date_time.strftime('%a %Y-%m-%d %H:%M:%S')


######################### Users ###########################

def get_user_teams(user):
    teams = Team.query.all()
    for team in teams:
        if user in get_team_members(team):
            yield team


def get_n_user_teams(user):
    return len(get_user_teams(user))


def get_hashed_password(plain_text_password):
    """Hash a password for the first time
    (Using bcrypt, the salt is saved into the hash itself)"""
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hased password. Using bcrypt, the salt is saved into the
    hash itself"""
    return bcrypt.checkpw(plain_text_password, hashed_password)


def create_user(name, password, lastname, firstname, email):
    hashed_password = get_hashed_password(password)
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=email)
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


def print_users():
    print('***************** List of users ****************')
    for user in User.query.order_by(User.id):
        print('{} belongs to teams:'.format(user))
        for team in get_user_teams(user):
            print('\t{}'.format(team))


######################### Teams ###########################


def get_team_members(team):
    if team.initiator is not None:
        # "yield from" in Python 3.3
        for member in get_team_members(team.initiator):
            yield member
        for member in get_team_members(team.acceptor):
            yield member
    else:
        yield team.admin


def get_n_team_members(team):
    return len(list(get_team_members(team)))


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
    if n_members_new > max_members_per_team:
        raise MergeTeamError(
            'Too big team: new team would be of size {}, the max is {}'.format(
                n_members_new, max_members_per_team))

    members_initiator = get_team_members(initiator)
    members_acceptor = get_team_members(acceptor)

    # Testing if team (same members) exists under a different name. If the
    # name is the same, we break. If the loop goes through, we add new team.
    members_set = set(members_initiator).union(set(members_acceptor))
    for team in Team.query:
        if members_set == set(get_team_members(team)):
            if name == team.name:
                break  # ok, but don't add new team, just set them to inactive
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
    return team


def print_active_teams():
    print('***************** List of active teams ****************')
    for team in Team.query.filter(Team.is_active):
        print('{} members:'.format(team))
        for member in get_team_members(team):
            print('\t{}'.format(member))


######################### Submissions ###########################


def make_submission(team_name, name, f_name_list):
    # TODO: to call unit tests on submitted files. Those that are found
    # in the table that describes the workflow. For the rest just check
    # maybe size
    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        name=name, team=team).one_or_none()
    cv_folds = CVFold.query.all()
    if submission is None:
        submission = Submission(name=name, team=team)
        # Adding submission files
        for f_name in f_name_list:
            submission_file = SubmissionFile(
                name=f_name, submission=submission)
            db.session.add(submission_file)
        db.session.add(submission)
        # Adding (empty) submission on cv folds
        for cv_fold in cv_folds:
            submission_on_cv_fold = SubmissionOnCVFold(
                submission=submission, cv_fold=cv_fold)
            db.session.add(submission_on_cv_fold)
        db.session.commit()
    else:
        # We allow resubmit for new or failing submissions
        if submission.state == 'new' or 'error' in submission.state:
            submission.state = 'new'
            # Updating existing files or adding new if not in list
            for f_name in f_name_list:
                submission_file = SubmissionFile.query.filter_by(
                    name=f_name, submission=submission).one_or_none()
                if submission_file is None:
                    submission_file = SubmissionFile(
                        name=f_name, submission=submission)
                    db.session.add(submission_file)
                # else copy the file should be there
                # right now we don't delete files that were not resubmitted,
                # allowing partial resubmission

            # Updating submission on cv folds
            submission_on_cv_folds = SubmissionOnCVFold.query.filter(
                SubmissionOnCVFold.submission == submission).all()
            for submission_on_cv_fold in submission_on_cv_folds:
                # couldn't figure out how to reset to default values
                db.session.delete(submission_on_cv_fold)
                db.session.add(submission_on_cv_fold)
            db.session.commit()
#            for cv_fold in cv_folds:
#                submission_on_cv_fold = SubmissionOnCVFold(
#                    submission=submission, cv_fold=cv_fold)
#                db.session.add(submission_on_cv_fold)
#            db.session.commit()
        else:
            raise DuplicateSubmissionError(
                'Submission "{}" of team "{}" exists already'.format(
                    name, team_name))

    # We should copy files here
    return submission


def train_test_submissions(force_retrain_test=False):
    for submission in Submission.query.all():
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

    for cv_fold in CVFold.query.all():
        cv_fold.compute_contributivity(force_ensemble)
    for submission in Submission.query.all():
        submission.set_contributivity()
    db.session.commit()


def get_public_leaderboard(team_name=None, user=None):
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
            Team.name == team_name).order_by(
            Submission.valid_score_cv_bag.desc()).all()
    elif user is not None:
        submissions_teams = []
        for team_name in [team.name for team in get_user_teams(user)]:
            submissions_teams += db.session.query(Submission, Team).filter(
                Team.id == Submission.team_id).filter(
                Team.name == team_name).filter(
                Submission.is_public_leaderboard).order_by(
                Submission.valid_score_cv_bag.desc()).all()
    else:
        submissions_teams = db.session.query(Submission, Team).filter(
            Team.id == Submission.team_id).filter(
            Submission.is_public_leaderboard).order_by(
            Submission.valid_score_cv_bag.desc()).all()
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
                      submission.name_with_link,
                      round(submission.valid_score_cv_bag, 2),
                      int(100 * submission.contributivity + 0.5),
                      int(submission.train_time_cv_mean + 0.5),
                      int(submission.valid_time_cv_mean + 0.5),
                      _date_time_format(submission.submission_timestamp)])}
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


def get_failed_submissions(user):
    """
    Returns
    -------
    leaderboard_html : html string
    """
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
                      _date_time_format(submission.submission_timestamp),
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


def print_submissions():
    print('***************** List of submissions ****************')
    for submission in Submission.query.order_by(Submission.id).all():
        print submission
        print('\tstate = {}'.format(submission.state))
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


def add_cv_folds(cv):
    for train_is, test_is in cv:
        cv_fold = CVFold(train_is=train_is, test_is=test_is)
        db.session.add(cv_fold)
    db.session.commit()
