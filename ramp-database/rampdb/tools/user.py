import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from ramputils.password import hash_password

from ..exceptions import NameClashError
from ..model import Team
from ..model import User

from ._query import select_team_by_name
from ._query import select_user_by_email
from ._query import select_user_by_name

logger = logging.getLogger('RAMP-DATABASE')


def create_user(session, name, password, lastname, firstname, email,
                access_level='user', hidden_notes='', linkedin_url='',
                twitter_url='', facebook_url='', google_url='', github_url='',
                website_url='', bio='', is_want_news=True):
    """Create a new user in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The username.
    password : str
        The password.
    lastname : str
        The user lastname.
    firstname : str
        The user firstname.
    email : str
        The user email address.
    access_level : {'admin', 'user', 'asked'}, default='user'
        The access level of the user.
    hidden_notes : str, default=''
        Some hidden notes.
    linkedin_url : str, default=''
        Linkedin URL.
    twitter_url : str, default=''
        Twitter URL.
    facebook_url : str, default=''
        Facebook URL.
    google_url : str, default=''
        Google URL.
    github_url : str, default=''
        GitHub URL.
    website_url : str, default=''
        Website URL.
    bio : str, default = ''
        User biography.
    is_want_news : bool, default is True
        User wish to receive some news.

    Returns
    -------
    user : :class:`rampdb.model.User`
        The user entry in the database.
    """
    # decode the hashed password (=bytes) because database columns is
    # String
    hashed_password = hash_password(password).decode()
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=email,
                access_level=access_level, hidden_notes=hidden_notes,
                linkedin_url=linkedin_url, twitter_url=twitter_url,
                facebook_url=facebook_url, google_url=google_url,
                github_url=github_url, website_url=website_url, bio=bio,
                is_want_news=is_want_news)

    # Creating default team with the same name as the user
    # user is admin of his/her own team
    team = Team(name=name, admin=user)
    session.add(team)
    session.add(user)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        message = ''
        try:
            select_user_by_name(session, name)
            message += 'username is already in use'
        except NoResultFound:
            # We only check for team names if username is not in db
            try:
                select_team_by_name(session, name)
                message += 'username is already in use as a team name'
            except NoResultFound:
                pass
        try:
            select_user_by_email(session, email)
            if message:
                message += ' and '
            message += 'email is already in use'
        except NoResultFound:
            pass
        if message:
            raise NameClashError(message)
        else:
            raise e
    logger.info('Creating {}'.format(user))
    logger.info('Creating {}'.format(team))
    return user


def approve_user(session, name):
    """Approve a user once it is created.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The name of the user.
    """
    user = select_user_by_name(session, name)
    if user.access_level == 'asked':
        user.access_level = 'user'
    user.is_authenticated = True
    session.commit()
    # TODO: be sure that we send an email
    # send_mail(user.email, 'RAMP sign-up approved', '')


def get_user_by_name(session, name):
    """Get a user by his/her name

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str or None
        The name of the user. If None, all users will be queried.

    Returns
    -------
    user : :class:`rampdb.model.User` or list of :class:`rampdb.model.User`
        The queried user.
    """
    return select_user_by_name(session, name)


def get_team_by_name(session, name):
    """Get a team by its name

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str or None
        The name of the team. If None, all teams will be queried.

    Returns
    -------
    team : :class:`rampdb.model.Team` or list of :class:`rampdb.model.Team`
        The queried team.
    """
    return select_team_by_name(session, name)
