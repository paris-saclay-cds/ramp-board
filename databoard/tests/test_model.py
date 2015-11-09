from databoard.config import set_engine_and_session, get_session, get_engine
set_engine_and_session('sqlite:///:memory:', echo=False)
session = get_session()
engine = get_engine()

import databoard.db.model as model
from databoard.db.model import Team, Submission, DBBase

DBBase.metadata.create_all(engine)


def test_password_hashing():
    plain_text_password = "hjst3789ep;ocikaqjw"
    hashed_password = model.get_hashed_password(plain_text_password)
    assert model.check_password(plain_text_password, hashed_password)
    assert not model.check_password("hjst3789ep;ocikaqji", hashed_password)


def test_create_user():
    model.create_user(
        name='kegl', password='bla', lastname='Kegl',
        firstname='Balazs', email='balazs.kegl@gmail.com')
    model.create_user(
        name='agramfort', password='bla', lastname='Gramfort',
        firstname='Alexandre', email='alexandre.gramfort@gmail.com')
    model.create_user(
        name='akazakci', password='bla', lastname='Akin',
        firstname='Kazakci', email='osmanakin@gmail.com')
    model.create_user(
        name='mcherti', password='bla', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com')

    try:
        model.create_user(
            name='kegl', password='bla', lastname='Kegl',
            firstname='Balazs', email='balazs.kegl@hotmail.com')
    except model.NameClashError as e:
        assert e.value == 'username is already in use'

    try:
        model.create_user(
            name='kegl_dupl_email', password='bla', lastname='Kegl',
            firstname='Balazs', email='balazs.kegl@gmail.com')
    except model.NameClashError as e:
        assert e.value == 'email is already in use'


def test_merge_teams():
    model.merge_teams(
        name='kemfort', initiator_name='kegl', acceptor_name='agramfort')
    model.merge_teams(
        name='mchezakci', initiator_name='mcherti', acceptor_name='akazakci')
    try:
        model.merge_teams(
            name='kemfezakci', initiator_name='kemfort',
            acceptor_name='mchezakci')
    except model.MergeTeamError as e:
        assert e.value == \
            'Too big team: new team would be of size 4, the max is 3'

    try:
        model.merge_teams(
            name='kezakci', initiator_name='kegl', acceptor_name='mchezakci')
    except model.MergeTeamError as e:
        assert e.value == 'Merge initiator is not active'
    try:
        model.merge_teams(
            name='mchezagl', initiator_name='mchezakci', acceptor_name='kegl')
    except model.MergeTeamError as e:
        assert e.value == 'Merge acceptor is not active'

    # simulating that in a new ramp single-user teams are active again, so
    # they can try to re-form eisting teams
    session.query(Team).filter_by(name='akazakci').one().is_active = True
    session.query(Team).filter_by(name='mcherti').one().is_active = True
    session.commit()
    try:
        model.merge_teams(
            name='akazarti', initiator_name='akazakci',
            acceptor_name='mcherti')
    except model.MergeTeamError as e:
        assert e.value == \
            'Team exists with the same members, team name = mchezakci'
    # but it should go through if name is the same (even if initiator and
    # acceptor are not the same)
    model.merge_teams(
        name='mchezakci', initiator_name='akazakci', acceptor_name='mcherti')

    session.query(Team).filter_by(name='akazakci').one().is_active = False
    session.query(Team).filter_by(name='mcherti').one().is_active = False
    session.commit()


def test_make_submission():
    model.make_submission('kemfort', 'rf', 'classifier.py')
    model.make_submission('mchezakci', 'rf', 'classifier.py')
    model.make_submission('kemfort', 'rf2', 'classifier.py')

    # resubmitting 'new' is OK
    model.make_submission('kemfort', 'rf', 'classifier.py')

    team = session.query(Team).filter_by(name='kemfort').one()
    submission = session.query(Submission).filter_by(
        team=team, name='rf').one()

    submission.trained_state = 'error'
    session.commit()
    # resubmitting 'error' is OK
    model.make_submission('kemfort', 'rf', 'classifier.py')

    submission.tested_state = 'error'
    session.commit()
    # resubmitting 'error' is OK
    model.make_submission('kemfort', 'rf', 'classifier.py')

    submission.trained_state = 'trained'
    session.commit()
    # resubmitting 'trained' is not OK
    try:
        model.make_submission('kemfort', 'rf', 'classifier.py')
    except model.DuplicateSubmissionError as e:
        assert e.value == 'Submission "rf" of team "kemfort" exists already'


def test_print_db():
    model.print_users()
    model.print_active_teams()
    model.print_submissions()


def test_leaderboard():
    team = session.query(Team).filter_by(name='kemfort').one()
    submissions = session.query(Submission).filter_by(team=team).all()
    for submission in submissions:
        submission.trained_state = 'scored'
    print model.get_public_leaderboard()

