from databoard.db.model import db, NameClashError, MergeTeamError
from databoard.db.model import Team, Submission
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


def test_make_submission():
    db_tools.make_submission('kemfort', 'rf', 'classifier.py')
    db_tools.make_submission('mchezakci', 'rf', 'classifier.py')
    db_tools.make_submission('kemfort', 'rf2', 'classifier.py')

    # resubmitting 'new' is OK
    db_tools.make_submission('kemfort', 'rf', 'classifier.py')

    team = db.session.query(Team).filter_by(name='kemfort').one()
    submission = db.session.query(Submission).filter_by(
        team=team, name='rf').one()

    submission.trained_state = 'error'
    db.session.commit()
    # resubmitting 'error' is OK
    db_tools.make_submission('kemfort', 'rf', 'classifier.py')

    submission.tested_state = 'error'
    db.session.commit()
    # resubmitting 'error' is OK
    db_tools.make_submission('kemfort', 'rf', 'classifier.py')

    submission.trained_state = 'trained'
    db.session.commit()
    # resubmitting 'trained' is not OK
    try:
        db_tools.make_submission('kemfort', 'rf', 'classifier.py')
    except db_tools.DuplicateSubmissionError as e:
        assert e.value == 'Submission "rf" of team "kemfort" exists already'


def test_print_db():
    db_tools.print_users()
    db_tools.print_active_teams()
    db_tools.print_submissions()


def test_leaderboard():
    team = db.session.query(Team).filter_by(name='kemfort').one()
    submissions = db.session.query(Submission).filter_by(team=team).all()
    for submission in submissions:
        submission.trained_state = 'scored'
    print db_tools.get_public_leaderboard()

