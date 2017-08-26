import os
from distutils.dir_util import mkpath

import databoard.config as config
from databoard import db
from databoard.model import NameClashError, MergeTeamError, User, Event
import databoard.db_tools as db_tools
from databoard.remove_test_db import recreate_test_db


def test_set_config_to_test():
    config.root_path = os.path.join('.')
    config.submissions_d_name = 'test_submissions'
    config.submissions_path = os.path.join(
        config.root_path, config.submissions_d_name)
    mkpath(os.path.join(config.problems_path, 'iris', 'data', 'public'))
    mkpath(os.path.join(config.problems_path, 'iris', 'data', 'private'))
    mkpath(os.path.join(
        config.problems_path, 'boston_housing', 'data', 'public'))
    mkpath(os.path.join(
        config.problems_path, 'boston_housing', 'data', 'private'))


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
    db_tools.add_keyword('botany', 'data_domain', 'scientific data', 'Botany.')
    db_tools.add_keyword(
        'real estate', 'data_domain', 'industrial data', 'Real estate.')
    db_tools.add_keyword(
        'regression', 'data_science_theme', None, 'Regression.')
    db_tools.add_keyword(
        'classification', 'data_science_theme', None, 'Classification.')
    db_tools.add_problem_keyword('iris', 'classification')
    db_tools.add_problem_keyword('iris', 'botany')
    db_tools.add_problem_keyword('boston_housing', 'regression')
    db_tools.add_problem_keyword('boston_housing', 'real estate')


def test_create_user():
    # db_tools.add_users_from_file('databoard/tests/data/users_to_add.csv')
    db_tools.create_user(
        name='kegl', password='wine fulcra kook homy',
        lastname='Kegl', firstname='Balazs',
        email='balazs.kegl@gmail.com', access_level='admin')
    db_tools.create_user(
        name='agramfort', password='Fushun helium pigeon radon',
        lastname='Gramfort', firstname='Alexandre',
        email='alexandre.gramfort@gmail.com', access_level='admin')
    db_tools.create_user(
        name='akazakci', password='Sept. bestir Ottawa seven',
        lastname='Akin', firstname='Kazakci',
        email='osmanakin@gmail.com', access_level='admin')
    db_tools.create_user(
        name='mcherti', password='blown ashcan manful dost', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com',
        access_level='admin')

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


def test_sign_up_team():
    db_tools.sign_up_team('iris_test', 'kegl')
    db_tools.sign_up_team('boston_housing_test', 'kegl')


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
    print(event.combined_combined_valid_score)
    print(event.combined_combined_valid_score_str)
    db_tools.make_submission_and_copy_files(
        'iris_test', 'kegl', 'starting_kit_test',
        'problems/iris/deposited_submissions/kegl/rf')

    db_tools.make_submission_and_copy_files(
        'iris_test', 'kegl', 'rf2',
        'problems/iris/deposited_submissions/kegl/rf2')

    # db_tools.make_submission_and_copy_files(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')
    # db_tools.make_submission_and_copy_files(
    #     'mchezakci', 'rf',
    #     'test_submissions/mchezakci/mfcee225579956cac0717ca38e7e4b529b28679cf')
    # db_tools.make_submission_and_copy_files(
    #     'kemfort', 'rf2',
    #     'test_submissions/kemfort/ma971cc83c886aaaad37d25029e7718c00ac3b4cd')
    # db_tools.make_submission_and_copy_files(
    #     'kemfort', 'training_error',
    #     'test_submissions/kemfort/mb5fa97067800c9c4c376b4d5beea3fd8a5db72c0')
    # db_tools.make_submission_and_copy_files(
    #     'kemfort', 'validating_error',
    #     'test_submissions/kemfort/mde194f09b58f6e519b334908862351138b302bd2')
    db_tools.print_submissions()

    # resubmitting 'new' is OK
    db_tools.make_submission_and_copy_files(
        'iris_test', 'kegl', 'rf2',
        'problems/iris/deposited_submissions/kegl/rf2')
    # db_tools.make_submission_and_copy_files(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')

    # team = Team.query.filter_by(name='kemfort').one()
    # submission = Submission.query.filter_by(team=team, name='rf').one()

    db_tools.set_state(
        'iris_test', 'kegl', 'starting_kit_test', 'training_error')
    # resubmitting 'error' is OK
    db_tools.make_submission_and_copy_files(
        'iris_test', 'kegl', 'rf',
        'problems/iris/deposited_submissions/kegl/rf')
    # db_tools.make_submission_and_copy_files(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')

    db_tools.set_state(
        'iris_test', 'kegl', 'starting_kit_test', 'testing_error')
    # resubmitting 'error' is OK
    db_tools.make_submission_and_copy_files(
        'iris_test', 'kegl', 'starting_kit_test',
        'problems/iris/deposited_submissions/kegl/rf')
    # db_tools.make_submission_and_copy_files(
    #     'kemfort', 'rf',
    #     'test_submissions/kemfort/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')

    db_tools.set_state('iris_test', 'kegl', 'starting_kit_test', 'trained')
    # resubmitting 'trained' is not OK
    try:
        db_tools.make_submission_and_copy_files(
            'iris_test', 'kegl', 'starting_kit_test',
            'problems/iris/deposited_submissions/kegl/rf')
        # db_tools.make_submission_and_copy_files(
        #     'kemfort', 'rf',
        #     'test_submissions/m3af2c986ca68d1598e93f653c0c0ae4b5e3449ae')
    except db_tools.DuplicateSubmissionError as e:
        assert e.value == 'Submission "starting_kit_test" of team "kegl" at event "iris_test" exists already'
        # assert e.value == 'Submission "rf" of team "kemfort" exists already'

    db_tools.set_state('iris_test', 'kegl', 'starting_kit_test', 'testing_error')

    db_tools.make_submission_and_copy_files(
        'boston_housing_test', 'kegl', 'starting_kit_test',
        'problems/boston_housing/deposited_submissions/kegl/rf')
    event = Event.query.filter_by(name='boston_housing_test').one()
    event.min_duration_between_submissions = 0
    db.session.commit()
    db_tools.make_submission_and_copy_files(
        'boston_housing_test', 'kegl', 'rf2',
        'problems/boston_housing/deposited_submissions/kegl/rf2')

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
    db_tools.compute_contributivity('iris_test')
    db_tools.compute_contributivity('boston_housing_test')


def test_print_db():
    db_tools.print_problems()
    print('\n')
    db_tools.print_events()
    print('\n')
    db_tools.print_users()
    print('\n')
    db_tools.print_active_teams(event_name='iris_test')
    print('\n')
    db_tools.print_submissions(event_name='iris_test')

    print('\n')
    db_tools.print_active_teams(event_name='boston_housing_test')
    print('\n')
    db_tools.print_submissions(event_name='boston_housing_test')


def test_leaderboard():
    current_user = User.query.filter_by(name='kegl').one()
    print('\n')
    print('***************** Leaderboard ****************')
    print(db_tools.get_leaderboards(
        'iris_test', user_name=current_user.name))
    print(db_tools.get_leaderboards(
        'boston_housing_test', user_name=current_user.name))
    print('***************** Private leaderboard ****************')
    print(db_tools.get_private_leaderboards('iris_test'))
    print(db_tools.get_private_leaderboards('boston_housing_test'))
    print('*********** Leaderboard of kegl ***********')
    print(db_tools.get_leaderboards(
            'iris_test', user_name='kegl'))
    print('*********** Private leaderboard of kegl ***********')
    print(db_tools.get_private_leaderboards('iris_test', user_name='kegl'))
    print('*********** Failing leaderboard of kegl ***********')
    print(db_tools.get_failed_leaderboard('iris_test', user_name='kegl'))
    print('*********** New leaderboard of kegl ***********')
    print(db_tools.get_new_leaderboard('iris_test', user_name='kegl'))
