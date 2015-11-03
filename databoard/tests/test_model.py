from ..db.model import get_hashed_password, check_password,\
    create_user, merge_teams, NameClashError, MergeTeamError, \
    User, Team, session


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
    merge_teams(name='kemfort', initiator_name='kegl', acceptor_name='agramfort')
    merge_teams(
        name='mchezakci', initiator_name='mcherti', acceptor_name='akazakci')
    try:
        merge_teams(
            name='kemfezakci', initiator_name='kemfort', acceptor_name='mchezakci')
    except MergeTeamError as e:
        assert e.value == 'Too big team: new team would be of size 4, the max is 3'

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

for user in session.query(User).order_by(User.user_id):
    print user, 'belongs to teams:'
    for team in user.get_teams():
        print '\t', team

for team in session.query(Team).order_by(Team.team_id):
    print team, 'members:'
    for member in team.get_members():
        print '\t', member
