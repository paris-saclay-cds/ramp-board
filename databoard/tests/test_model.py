from ..config import set_engine_and_session, get_session
set_engine_and_session('sqlite:///:memory:', echo=False)
session = get_session()

from ..db.model import get_hashed_password, check_password,\
    create_user, merge_teams, NameClashError, MergeTeamError,\
    DuplicateSubmissionError,\
    User, Team, Submission, make_submission,\
    print_users, print_active_teams, print_submissions


def test_password_hashing():
    plain_text_password = "hjst3789ep;ocikaqjw"
    hashed_password = get_hashed_password(plain_text_password)
    assert check_password(plain_text_password, hashed_password)
    assert not check_password("hjst3789ep;ocikaqji", hashed_password)


def test_create_user():
    create_user(name='kegl', password='bla',
                lastname='Kegl', firstname='Balazs',
                email='balazs.kegl@gmail.com')
    create_user(name='agramfort', password='bla',
                lastname='Gramfort', firstname='Alexandre',
                email='alexandre.gramfort@gmail.com')
    create_user(name='akazakci', password='bla',
                lastname='Akin', firstname='Kazakci',
                email='osmanakin@gmail.com')
    create_user(name='mcherti', password='bla',
                lastname='Cherti', firstname='Mehdi',
                email='mehdicherti@gmail.com')

    try:
        create_user(name='kegl', password='bla',
                    lastname='Kegl', firstname='Balazs',
                    email='balazs.kegl@hotmail.com')
    except NameClashError as e:
        assert e.value == 'username is already in use'

    try:
        create_user(name='kegl_dupl_email', password='bla',
                    lastname='Kegl', firstname='Balazs',
                    email='balazs.kegl@gmail.com')
    except NameClashError as e:
        assert e.value == 'email is already in use'


def test_merge_teams():
    merge_teams(
        name='kemfort', initiator_name='kegl', acceptor_name='agramfort')
    merge_teams(
        name='mchezakci', initiator_name='mcherti', acceptor_name='akazakci')
    try:
        merge_teams(name='kemfezakci', initiator_name='kemfort',
                    acceptor_name='mchezakci')
    except MergeTeamError as e:
        assert e.value == \
            'Too big team: new team would be of size 4, the max is 3'

    try:
        merge_teams(
            name='kezakci', initiator_name='kegl', acceptor_name='mchezakci')
    except MergeTeamError as e:
        assert e.value == 'Merge initiator is not active'
    try:
        merge_teams(
            name='mchezagl', initiator_name='mchezakci', acceptor_name='kegl')
    except MergeTeamError as e:
        assert e.value == 'Merge acceptor is not active'

    # simulating that in a new ramp single-user teams are active again, so
    # they can try to re-form eisting teams
    session.query(Team).filter_by(name='akazakci').one().is_active = True
    session.query(Team).filter_by(name='mcherti').one().is_active = True
    session.commit()
    try:
        merge_teams(
            name='akazarti', initiator_name='akazakci',
            acceptor_name='mcherti')
    except MergeTeamError as e:
        assert e.value == \
            'Team exists with the same members, team name = mchezakci'
    # but it should go through if name is the same (even if initiator and
    # acceptor are not the same)
    merge_teams(
        name='mchezakci', initiator_name='akazakci', acceptor_name='mcherti')

    session.query(Team).filter_by(name='akazakci').one().is_active = False
    session.query(Team).filter_by(name='mcherti').one().is_active = False
    session.commit()


def test_make_submission():
    make_submission('kemfort', 'rf', 'classifier.py')
    make_submission('mchezakci', 'rf', 'classifier.py')
    make_submission('kemfort', 'rf2', 'classifier.py')

    # resubmitting 'new' is OK
    make_submission('kemfort', 'rf', 'classifier.py')

    team = session.query(Team).filter_by(name='kemfort').one()
    submission = session.query(Submission).filter_by(
        team=team, name='rf').one()

    submission.trained_state = 'error'
    session.commit()
    # resubmitting 'error' is OK
    make_submission('kemfort', 'rf', 'classifier.py')

    submission.tested_state = 'error'
    session.commit()
    # resubmitting 'error' is OK
    make_submission('kemfort', 'rf', 'classifier.py')

    submission.trained_state = 'trained'
    session.commit()
    # resubmitting 'trained' is not OK
    try:
        make_submission('kemfort', 'rf', 'classifier.py')
    except DuplicateSubmissionError as e:
        assert e.value == 'Submission "rf" of team "kemfort" exists already'


def test_print_db():
    print_users()
    print_active_teams()
    print_submissions()
