import os
import databoard.config as config
from numpy.testing import assert_array_equal
from databoard.db.model import db, NameClashError, MergeTeamError,\
    MissingSubmissionFile
from databoard.db.model import Team, Submission, CVFold
import databoard.db.tools as db_tools


from databoard.db.remove_test_db import recreate_test_db


def test_recreate_test_db():
    recreate_test_db()


def test_password_hashing():
    plain_text_password = "hjst3789ep;ocikaqjw"
    hashed_password = db_tools.get_hashed_password(plain_text_password)
    assert db_tools.check_password(plain_text_password, hashed_password)
    assert not db_tools.check_password("hjst3789ep;ocikaqji", hashed_password)


def test_create_user():
    db_tools.create_user(
        name='kegl', password='bla', lastname='Kegl',
        firstname='Balazs', email='balazs.kegl@gmail.com')
    db_tools.create_user(
        name='agramfort', password='bla', lastname='Gramfort',
        firstname='Alexandre', email='alexandre.gramfort@gmail.com')
    db_tools.create_user(
        name='akazakci', password='bla', lastname='Akin',
        firstname='Kazakci', email='osmanakin@gmail.com')
    db_tools.create_user(
        name='mcherti', password='bla', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com')

    try:
        db_tools.create_user(
            name='kegl', password='bla', lastname='Kegl',
            firstname='Balazs', email='balazs.kegl@hotmail.com')
    except NameClashError as e:
        assert e.value == 'username is already in use'

    try:
        db_tools.create_user(
            name='kegl_dupl_email', password='bla', lastname='Kegl',
            firstname='Balazs', email='balazs.kegl@gmail.com')
    except NameClashError as e:
        assert e.value == 'email is already in use'


def test_merge_teams():
    db_tools.merge_teams(
        name='kemfort', initiator_name='kegl', acceptor_name='agramfort')
    db_tools.merge_teams(
        name='mchezakci', initiator_name='mcherti', acceptor_name='akazakci')
    try:
        db_tools.merge_teams(
            name='kemfezakci', initiator_name='kemfort',
            acceptor_name='mchezakci')
    except MergeTeamError as e:
        assert e.value == \
            'Too big team: new team would be of size 4, the max is 3'

    try:
        db_tools.merge_teams(
            name='kezakci', initiator_name='kegl', acceptor_name='mchezakci')
    except MergeTeamError as e:
        assert e.value == 'Merge initiator is not active'
    try:
        db_tools.merge_teams(
            name='mchezagl', initiator_name='mchezakci', acceptor_name='kegl')
    except MergeTeamError as e:
        assert e.value == 'Merge acceptor is not active'

    # simulating that in a new ramp single-user teams are active again, so
    # they can try to re-form eisting teams
    db.session.query(Team).filter_by(name='akazakci').one().is_active = True
    db.session.query(Team).filter_by(name='mcherti').one().is_active = True
    db.session.commit()
    try:
        db_tools.merge_teams(
            name='akazarti', initiator_name='akazakci',
            acceptor_name='mcherti')
    except MergeTeamError as e:
        assert e.value == \
            'Team exists with the same members, team name = mchezakci'
    # but it should go through if name is the same (even if initiator and
    # acceptor are not the same)
    db_tools.merge_teams(
        name='mchezakci', initiator_name='akazakci', acceptor_name='mcherti')

    db.session.query(Team).filter_by(name='akazakci').one().is_active = False
    db.session.query(Team).filter_by(name='mcherti').one().is_active = False
    db.session.commit()


def test_set_config_to_test():
    config.config_object.ramp_name = 'iris'
    origin_path = os.path.join('ramps', config.config_object.ramp_name)
    config.root_path = os.path.join('.')
    tests_path = os.path.join('databoard', 'tests')

    config.raw_data_path = os.path.join(origin_path, 'data', 'raw')
    config.public_data_path = os.path.join(tests_path, 'data', 'public')
    config.private_data_path = os.path.join(tests_path, 'data', 'private')
    config.submissions_d_name = 'test_submissions'
    config.submissions_path = os.path.join(
        config.root_path, config.submissions_d_name)
    config.deposited_submissions_path = os.path.join(
        origin_path, 'deposited_submissions')
    config.config_object.n_cpus = 3


def test_add_cv_folds():
    specific = config.config_object.specific
    specific.prepare_data()
    _, y_train = specific.get_train_data()
    cv = specific.get_cv(y_train)
    db_tools.add_cv_folds(cv)
    cv_folds = db.session.query(CVFold)
    for cv_fold_1, cv_fold_2 in zip(cv, cv_folds):
        train_is, test_is = cv_fold_1
        # print cv_fold_1
        # print cv_fold_2
        assert_array_equal(train_is, cv_fold_2.train_is)
        assert_array_equal(test_is, cv_fold_2.test_is)


def test_make_submission():
    db_tools.make_submission('kemfort', 'rf', ['classifier.py'])
    db_tools.make_submission('mchezakci', 'rf', ['classifier.py'])
    db_tools.make_submission('kemfort', 'rf2', ['classifier.py'])
    db_tools.print_submissions()

    # resubmitting 'new' is OK
    db_tools.make_submission('kemfort', 'rf', ['classifier.py'])

    team = db.session.query(Team).filter_by(name='kemfort').one()
    submission = db.session.query(Submission).filter_by(
        team=team, name='rf').one()

    submission.state = 'training_error'
    db.session.commit()
    # resubmitting 'error' is OK
    db_tools.make_submission(
        'kemfort', 'rf', ['classifier.py'])

    submission.state = 'testing_error'
    db.session.commit()
    # resubmitting 'error' is OK
    db_tools.make_submission(
        'kemfort', 'rf', ['classifier.py'])

    submission.state = 'trained'
    db.session.commit()
    # resubmitting 'trained' is not OK
    try:
        db_tools.make_submission('kemfort', 'rf', ['classifier.py'])
    except db_tools.DuplicateSubmissionError as e:
        assert e.value == 'Submission "rf" of team "kemfort" exists already'

    submission.state = 'testing_error'
    db.session.commit()
    # submitting non-existing file
    try:
        db_tools.make_submission('kemfort', 'rf', ['feature_extractor.py'])
    except MissingSubmissionFile as e:
        assert e.value == 'kemfort/rf/feature_extractor.py: ./test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae/feature_extractor.py'


# TODO: test all kinds of error states
def train_test_submissions():
    config.is_parallelize = False
    db_tools.train_test_submissions()
    db_tools.train_test_submissions()
    db_tools.train_test_submissions(force_retrain_test=True)
    config.is_parallelize = True
    db_tools.train_test_submissions(force_retrain_test=True)


def test_compute_contributivity():
    db_tools.compute_contributivity()


def test_print_db():
    print '\n'
    db_tools.print_users()
    print '\n'
    db_tools.print_active_teams()
    print '\n'
    db_tools.print_submissions()


def test_leaderboard():
    print '\n'
    print('***************** Leaderboard ****************')
    print db_tools.get_public_leaderboard()
