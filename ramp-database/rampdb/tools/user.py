import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from ramputils.password import hash_password

from ..exceptions import NameClashError
from ..model import Team
from ..model import User
from ..utils import setup_db

logger = logging.getLogger('DATABASE')


def create_user(config, name, password, lastname, firstname, email,
                access_level='user', hidden_notes='', linkedin_url='',
                twitter_url='', facebook_url='', google_url='', github_url='',
                website_url='', bio='', is_want_news=True):
    """Create a new user in the database.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
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
    user : rampdb.model.User
        The user entry in the database.
    """
    db, Session = setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
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
                session.query(User).filter(User.name == name).one()
                message += 'username is already in use'
            except NoResultFound:
                # We only check for team names if username is not in db
                try:
                    session.query(Team).filter(Team.name == name).one()
                    message += 'username is already in use as a team name'
                except NoResultFound:
                    pass
            try:
                session.query(User).filter(User.email == email).one()
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
