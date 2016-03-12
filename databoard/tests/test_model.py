import os

import databoard.config as config
from numpy.testing import assert_array_equal
from databoard import db
from databoard.model import NameClashError, MergeTeamError,\
    Team, Submission, CVFold, User, Event, EventTeam
import databoard.db_tools as db_tools


from databoard.remove_test_db import recreate_test_db


def test_set_config_to_test():
    config.min_duration_between_submissions = 0
    config.config_object.ramp_name = 'iris'
    origin_path = os.path.join('ramps', config.config_object.ramp_name)
    config.root_path = os.path.join('.')
    tests_path = os.path.join('databoard', 'tests')

    #config.raw_data_path = os.path.join(origin_path, 'data', 'raw')
    #config.public_data_path = os.path.join(tests_path, 'data', 'public')
    #config.private_data_path = os.path.join(tests_path, 'data', 'private')
    config.submissions_d_name = 'test_submissions'
    config.submissions_path = os.path.join(
        config.root_path, config.submissions_d_name)
    config.deposited_submissions_path = os.path.join(
        origin_path, config.deposited_submissions_d_name)
    config.sandbox_path = os.path.join(
        origin_path, config.sandbox_d_name)
    config.config_object.n_cpus = 3
    config.is_send_trained_mails = False


def test_recreate_test_db():
    recreate_test_db()


def test_password_hashing():
    plain_text_password = "hjst3789ep;ocikaqjw"
    hashed_password = db_tools.get_hashed_password(plain_text_password)
    assert db_tools.check_password(plain_text_password, hashed_password)
    assert not db_tools.check_password("hjst3789ep;ocikaqji", hashed_password)


def test_setup_problem():
    db_tools.setup_workflows()
    db_tools.add_problem('iris')
    db_tools.add_event('iris_test')
    db_tools.add_problem('boston_housing')
    db_tools.add_event('boston_housing_test')


def test_create_user():
    # db_tools.add_users_from_file('databoard/tests/data/users_to_add.csv')
    db_tools.create_user(
        name='kegl', password='wine fulcra kook homy', lastname='Kegl',
        firstname='Balazs', email='balazs.kegl@gmail.com',
        access_level='admin')
    db_tools.create_user(
        name='agramfort', password='bla', lastname='Gramfort',
        firstname='Alexandre', email='alexandre.gramfort@gmail.com',
        access_level='admin')
    db_tools.create_user(
        name='akazakci', password='bla', lastname='Akin',
        firstname='Kazakci', email='osmanakin@gmail.com',
        access_level='user')
    db_tools.create_user(
        name='mcherti', password='bla', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com',
        access_level='admin')
    db_tools.create_user(
        name='djabbz', password='bla', lastname='Benbouzid',
        firstname='Djalel', email='djalel.benbouzid@gmail.com',
        access_level='user')

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


def test_signup_team():
    db_tools.signup_team('iris_test', 'kegl')
    db_tools.signup_team('boston_housing_test', 'kegl')


# for now this is not functional, we should think through how teams
# should be merged when we have multiple RAMP events.
# def test_merge_teams():
#     db_tools.merge_teams(
#         name='kemfort', initiator_name='kegl', acceptor_name='agramfort')
#     db_tools.merge_teams(
#         name='mchezakci', initiator_name='mcherti', acceptor_name='akazakci')
#     try:
#         db_tools.merge_teams(
#             name='kemfezakci', initiator_name='kemfort',
#             acceptor_name='mchezakci')
#     except MergeTeamError as e:
#         assert e.value == \
#             'Too big team: new team would be of size 4, the max is 3'

#     try:
#         db_tools.merge_teams(
#             name='kezakci', initiator_name='kegl', acceptor_name='mchezakci')
#     except MergeTeamError as e:
#         assert e.value == 'Merge initiator is not active'
#     try:
#         db_tools.merge_teams(
#             name='mchezagl', initiator_name='mchezakci', acceptor_name='kegl')
#     except MergeTeamError as e:
#         assert e.value == 'Merge acceptor is not active'

#     # simulating that in a new ramp single-user teams are active again, so
#     # they can try to re-form eisting teams
#     Team.query.filter_by(name='akazakci').one().is_active = True
#     Team.query.filter_by(name='mcherti').one().is_active = True
#     db.session.commit()
#     try:
#         db_tools.merge_teams(
#             name='akazarti', initiator_name='akazakci',
#             acceptor_name='mcherti')
#     except MergeTeamError as e:
#         assert e.value == \
#             'Team exists with the same members, team name = mchezakci'
#     # but it should go through if name is the same (even if initiator and
#     # acceptor are not the same)
#     db_tools.merge_teams(
#         name='mchezakci', initiator_name='akazakci', acceptor_name='mcherti')

#     Team.query.filter_by(name='akazakci').one().is_active = False
#     Team.query.filter_by(name='mcherti').one().is_active = False
#     db.session.commit()


def test_make_submission():
    event = Event.query.filter_by(name='iris_test').one()
    team = Team.query.filter_by(name='kegl').one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()

    db_tools.make_submission(
        'iris_test', 'kegl', 'rf',
        'ramps/iris/deposited_submissions/kegl/rf')
    try:
        db_tools.make_submission(
            'iris_test', 'kegl', 'rf2',
            'ramps/iris/deposited_submissions/kegl/rf2')
    except db_tools.TooEarlySubmissionError as e:
        assert e.value == 'You need to wait 899 more seconds until next submission'
    event.min_duration_between_submissions = 0
    db.session.commit()
    db_tools.make_submission(
        'iris_test', 'kegl', 'rf2',
        'ramps/iris/deposited_submissions/kegl/rf2')


    # db_tools.make_submission(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')
    # db_tools.make_submission(
    #     'mchezakci', 'rf',
    #     'test_submissions/mchezakci/mfcee225579956cac0717ca38e7e4b529b28679cf')
    # db_tools.make_submission(
    #     'kemfort', 'rf2',
    #     'test_submissions/kemfort/ma971cc83c886aaaad37d25029e7718c00ac3b4cd')
    # db_tools.make_submission(
    #     'kemfort', 'training_error',
    #     'test_submissions/kemfort/mb5fa97067800c9c4c376b4d5beea3fd8a5db72c0')
    # db_tools.make_submission(
    #     'kemfort', 'validating_error',
    #     'test_submissions/kemfort/mde194f09b58f6e519b334908862351138b302bd2')
    db_tools.print_submissions()

    # resubmitting 'new' is OK
    db_tools.make_submission(
        'iris_test', 'kegl', 'rf2',
        'ramps/iris/deposited_submissions/kegl/rf')
    # db_tools.make_submission(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')

    submission = Submission.query.filter_by(
        event_team=event_team, name='rf').one()
    # team = Team.query.filter_by(name='kemfort').one()
    # submission = Submission.query.filter_by(team=team, name='rf').one()

    submission.state = 'training_error'
    db.session.commit()
    # resubmitting 'error' is OK
    db_tools.make_submission(
        'iris_test', 'kegl', 'rf',
        'ramps/iris/deposited_submissions/kegl/rf')
    # db_tools.make_submission(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')

    submission.state = 'testing_error'
    db.session.commit()
    # resubmitting 'error' is OK
    db_tools.make_submission(
        'iris_test', 'kegl', 'rf',
        'ramps/iris/deposited_submissions/kegl/rf')
    # db_tools.make_submission(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')

    submission.state = 'trained'
    db.session.commit()
    # resubmitting 'trained' is not OK
    try:
        db_tools.make_submission(
            'iris_test', 'kegl', 'rf', 
            'ramps/iris/deposited_submissions/kegl/rf')
        # db_tools.make_submission(
        #     'kemfort', 'rf',
        #     'test_submissions/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')
    except db_tools.DuplicateSubmissionError as e:
        assert e.value == 'Submission "rf" of team "kegl" at event "iris_test" exists already'
        # assert e.value == 'Submission "rf" of team "kemfort" exists already'

    submission.state = 'testing_error'

    db_tools.make_submission(
        'boston_housing_test', 'kegl', 'rf',
        'ramps/boston_housing/deposited_submissions/kegl/rf')
    event = Event.query.filter_by(name='boston_housing_test').one()
    event.min_duration_between_submissions = 0
    db.session.commit()
    db_tools.make_submission(
        'boston_housing_test', 'kegl', 'rf2',
        'ramps/boston_housing/deposited_submissions/kegl/rf2')

    db.session.commit()


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
    print('*********** Leaderboard of kemfort ***********')
    print db_tools.get_public_leaderboard(team_name='kemfort')
    print('*********** Leaderboard of kegl ***********')
    print db_tools.get_public_leaderboard(
        user=User.query.filter_by(name='kegl').one())
    print('*********** Failing submissions of kegl ***********')
    print db_tools.get_failed_submissions(
        user=User.query.filter_by(name='kegl').one())
