from ..db.model import get_hashed_password, check_password,\
    create_user, merge_teams, NameClashError, MergeTeamError, \
    User, Team, session

plain_text_password = "hjst3789ep;ocikaqjw"
hashed_password = get_hashed_password(plain_text_password)
assert check_password(plain_text_password, hashed_password)
assert not check_password("hjst3789ep;ocikaqji", hashed_password)

create_user(name='kegl', password='bla',
            family_name='Kegl', given_name='Balazs',
            email='balazs.kegl@gmail.com')
create_user(name='agramfort', password='bla',
            family_name='Gramfort', given_name='Alexandre',
            email='alexandre.gramfort@gmail.com')
create_user(name='akazakci', password='bla',
            family_name='Akin', given_name='Kazakci',
            email='osmanakin@gmail.com')
create_user(name='mcherti', password='bla',
            family_name='Cherti', given_name='Mehdi',
            email='mehdicherti@gmail.com')

try:
    create_user(name='kegl', password='bla',
                family_name='Kegl', given_name='Balazs',
                email='balazs.kegl@hotmail.com')
except NameClashError as e:
    assert e.value == 'username is already in use'

try:
    create_user(name='kegl_dupl_email', password='bla',
                family_name='Kegl', given_name='Balazs',
                email='balazs.kegl@gmail.com')
except NameClashError as e:
    assert e.value == 'email is already in use'

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
    for team in user.teams:
        print '\t', team

for team in session.query(Team).order_by(Team.team_id):
    print team, 'members:'
    for member in team.members:
        print '\t', member
