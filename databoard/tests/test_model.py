from databoard.config import set_engine_and_session, get_session, get_engine
set_engine_and_session('sqlite:///:memory:', echo=False)
session = get_session()
engine = get_engine()

from databoard.db.model_base import DBBase, NameClashError
import databoard.db.teams as teams
import databoard.db.users as users
import databoard.db.submissions as submissions
from databoard.db.submissions import Submission
from databoard.db.teams import Team

DBBase.metadata.create_all(engine)


def test_password_hashing():
    plain_text_password = "hjst3789ep;ocikaqjw"
    hashed_password = users.get_hashed_password(plain_text_password)
    assert users.check_password(plain_text_password, hashed_password)
    assert not users.check_password("hjst3789ep;ocikaqji", hashed_password)


def test_create_user():
    users.create_user(
        name='kegl', password='bla', lastname='Kegl',
        firstname='Balazs', email='balazs.kegl@gmail.com')
    users.create_user(
        name='agramfort', password='bla', lastname='Gramfort',
        firstname='Alexandre', email='alexandre.gramfort@gmail.com')
    users.create_user(
        name='akazakci', password='bla', lastname='Akin',
        firstname='Kazakci', email='osmanakin@gmail.com')
    users.create_user(
        name='mcherti', password='bla', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com')

    try:
        users.create_user(
            name='kegl', password='bla', lastname='Kegl',
            firstname='Balazs', email='balazs.kegl@hotmail.com')
    except NameClashError as e:
        assert e.value == 'username is already in use'

    try:
        users.create_user(
            name='kegl_dupl_email', password='bla', lastname='Kegl',
            firstname='Balazs', email='balazs.kegl@gmail.com')
    except NameClashError as e:
        assert e.value == 'email is already in use'


def test_merge_teams():
    teams.merge_teams(
        name='kemfort', initiator_name='kegl', acceptor_name='agramfort')
    teams.merge_teams(
        name='mchezakci', initiator_name='mcherti', acceptor_name='akazakci')
    try:
        teams.merge_teams(
            name='kemfezakci', initiator_name='kemfort',
            acceptor_name='mchezakci')
    except teams.MergeTeamError as e:
        assert e.value == \
            'Too big team: new team would be of size 4, the max is 3'

    try:
        teams.merge_teams(
            name='kezakci', initiator_name='kegl', acceptor_name='mchezakci')
    except teams.MergeTeamError as e:
        assert e.value == 'Merge initiator is not active'
    try:
        teams.merge_teams(
            name='mchezagl', initiator_name='mchezakci', acceptor_name='kegl')
    except teams.MergeTeamError as e:
        assert e.value == 'Merge acceptor is not active'

    # simulating that in a new ramp single-user teams are active again, so
    # they can try to re-form eisting teams
    session.query(Team).filter_by(name='akazakci').one().is_active = True
    session.query(Team).filter_by(name='mcherti').one().is_active = True
    session.commit()
    try:
        teams.merge_teams(
            name='akazarti', initiator_name='akazakci',
            acceptor_name='mcherti')
    except teams.MergeTeamError as e:
        assert e.value == \
            'Team exists with the same members, team name = mchezakci'
    # but it should go through if name is the same (even if initiator and
    # acceptor are not the same)
    teams.merge_teams(
        name='mchezakci', initiator_name='akazakci', acceptor_name='mcherti')

    session.query(Team).filter_by(name='akazakci').one().is_active = False
    session.query(Team).filter_by(name='mcherti').one().is_active = False
    session.commit()


def test_make_submission():
    submissions.make_submission('kemfort', 'rf', 'classifier.py')
    submissions.make_submission('mchezakci', 'rf', 'classifier.py')
    submissions.make_submission('kemfort', 'rf2', 'classifier.py')

    # resubmitting 'new' is OK
    submissions.make_submission('kemfort', 'rf', 'classifier.py')

    team = session.query(Team).filter_by(name='kemfort').one()
    submission = session.query(Submission).filter_by(
        team=team, name='rf').one()

    submission.trained_state = 'error'
    session.commit()
    # resubmitting 'error' is OK
    submissions.make_submission('kemfort', 'rf', 'classifier.py')

    submission.tested_state = 'error'
    session.commit()
    # resubmitting 'error' is OK
    submissions.make_submission('kemfort', 'rf', 'classifier.py')

    submission.trained_state = 'trained'
    session.commit()
    # resubmitting 'trained' is not OK
    try:
        submissions.make_submission('kemfort', 'rf', 'classifier.py')
    except submissions.DuplicateSubmissionError as e:
        assert e.value == 'Submission "rf" of team "kemfort" exists already'


def test_print_db():
    users.print_users()
    teams.print_active_teams()
    submissions.print_submissions()


def test_leaderboard():
    team = session.query(Team).filter_by(name='kemfort').one()
    submissions_ = session.query(Submission).filter_by(team=team).all()
    for submission in submissions_:
        submission.trained_state = 'scored'
    print submissions.get_public_leaderboard()
