from collections import defaultdict
import logging

import pandas as pd

from sqlalchemy.exc import IntegrityError

from ..exceptions import NameClashError
from ..model import Team
from ..model import User
from ..model import UserInteraction
from ..utils import hash_password

from ._query import select_team_by_name
from ._query import select_user_by_email
from ._query import select_user_by_name

logger = logging.getLogger('RAMP-DATABASE')


def add_user(session, name, password, lastname, firstname, email,
             access_level='user', hidden_notes='', linkedin_url='',
             twitter_url='', facebook_url='', google_url='', github_url='',
             website_url='', bio='', is_want_news=True):
    """Add a new user in the database.

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
    user : :class:`ramp_database.model.User`
        The user entry in the database.
    """
    # decode the hashed password (=bytes) because database columns is
    # String
    hashed_password = hash_password(password).decode()
    lower_case_email = email.lower()
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=lower_case_email,
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
        if select_user_by_name(session, name) is not None:
            message += 'username is already in use'
        elif select_team_by_name(session, name) is not None:
            # We only check for team names if username is not in db
            message += 'username is already in use as a team name'
        if select_user_by_email(session, lower_case_email) is not None:
            if message:
                message += ' and '
            message += 'email is already in use'
        if message:
            raise NameClashError(message)
        else:
            raise e
    logger.info('Creating {}'.format(user))
    logger.info('Creating {}'.format(team))
    return user


def delete_user(session, name):
    """Delete a user from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The name of the user.
    """
    user = session.query(User).filter(User.name == name).one()
    session.delete(user)
    session.commit()


def make_user_admin(session, name):
    """Make a user a RAMP admin.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The name of the user.
    """
    user = select_user_by_name(session, name)
    user.access_level = 'admin'
    user.is_authenticated = True
    session.commit()


def set_user_access_level(session, name, access_level="user"):
    """Change the access level of a user.

    In addition, it will also set the user as authenticated.

    Paramaters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The name of the user.
    access_level : {"asked", "user", admin}, default="user"
        User's access level
    """
    user = select_user_by_name(session, name)
    user.access_level = access_level
    user.is_authenticated = True
    session.commit()


def add_user_interaction(session, interaction=None, user=None, problem=None,
                         event=None, ip=None, note=None, submission=None,
                         submission_file=None, diff=None, similarity=None):
    """Add a user interaction in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    interactions : None or str, default is None
        The type of interaction.
    user : None or :class:`ramp_database.model.User`, default is None
        The user instance.
    problem : None or :class:`ramp_database.model.Problem`, default is None
        The problem instance.
    event : None or :class:`ramp_database.model.Event`, default is None
        The event instance.
    ip : None or str, default is None
        The ip address from the server.
    note : None or str, default is None
        Some notes.
    submission : None or :class:`ramp_database.model.Submission`, \
default is None
        The submission instance.
    submission_file : None or :class:`ramp_database.model.SubmissionFile`, \
default is None
        The submission file instance.
    diff : None or str, default is None
        The difference between two submissions.
    similarity : None or float, default is None
        The similarity of the submission.
    """
    user_interaction = UserInteraction(
        session=session, interaction=interaction, user=user, problem=problem,
        ip=ip, note=note, submission=submission, event=event,
        submission_file=submission_file, diff=diff, similarity=similarity
    )
    session.add(user_interaction)
    session.commit()


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


def get_user_by_name(session, name):
    """Get a user by his/her name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str or None
        The name of the user. If None, all users will be queried.

    Returns
    -------
    user : :class:`ramp_database.model.User` or list of \
:class:`ramp_database.model.User`
        The queried user.
    """
    return select_user_by_name(session, name)


def get_user_by_name_or_email(session, name):
    """Get a user by his/her name or email. It will return true if either
    is correct

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str or None
        The name of the user. If None, all users will be queried.

    Returns
    -------
    user : :class:`ramp_database.model.User` or list of \
:class:`ramp_database.model.User`
        The queried user.
    """
    return (select_user_by_email(session, name) or
            select_user_by_name(session, name))


def get_team_by_name(session, name):
    """Get a team by its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str or None
        The name of the team. If None, all teams will be queried.

    Returns
    -------
    team : :class:`ramp_database.model.Team` or list of \
:class:`ramp_database.model.Team`
        The queried team.
    """
    return select_team_by_name(session, name)


def get_user_interactions_by_name(session, name=None,
                                  output_format='dataframe'):
    """Get the user interactions.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str or None, default is None
        The name of the user. By default all interactions are returned.
    output_format : {'dataframe', 'html'}
        The output format to be returned.

    Returns
    -------
    user_interactions : output_format
        The user interactions queried and return as specified by
        ``output_format``.

    """
    user_interactions = session.query(UserInteraction)
    if name is None:
        user_interactions = user_interactions.all()
    else:
        user_interactions = \
            (user_interactions.filter(UserInteraction.user_id == User.id)
                              .filter(User.name == name)
                              .all())

    map_columns_attributes = defaultdict(list)
    for ui in user_interactions:
        map_columns_attributes['timestamp (UTC)'].append(ui.timestamp)
        map_columns_attributes['IP'].append(ui.ip)
        map_columns_attributes['interaction'].append(ui.interaction)
        map_columns_attributes['user'].append(getattr(ui.user, 'name', None))
        map_columns_attributes['event'].append(getattr(
            getattr(ui.event_team, 'event', None), 'name', None))
        map_columns_attributes['team'].append(getattr(
            getattr(ui.event_team, 'team', None), 'name', None))
        map_columns_attributes['submission_id'].append(ui.submission_id)
        map_columns_attributes['submission'].append(
            getattr(ui.submission, 'name_with_link', None))
        map_columns_attributes['file'].append(
            getattr(ui.submission_file, 'name_with_link', None))
        map_columns_attributes['code similarity'].append(
            ui.submission_file_similarity)
        map_columns_attributes['diff'].append(
            None if ui.submission_file_diff is None
            else '<a href="{}">diff</a>'.format(
                ui.submission_file_diff))
    df = (pd.DataFrame(map_columns_attributes)
            .sort_values('timestamp (UTC)', ascending=False)
            .set_index('timestamp (UTC)'))
    if output_format == 'html':
        return df.to_html(escape=False, index=False, max_cols=None,
                          max_rows=None, justify='left')
    return df


def set_user_by_instance(session, user, lastname, firstname, email,
                         linkedin_url='', twitter_url='', facebook_url='',
                         google_url='', github_url='', website_url='', bio='',
                         is_want_news=True):
    """Set the information of a user.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    user : :class:`ramp_database.model.User`
        The user instance to update.
    lastname : str
        The user lastname.
    firstname : str
        The user firstname.
    email : str
        The user email address.
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
    """
    logger.info('Update the profile of "{}"'.format(user))

    for field in ('lastname', 'firstname', 'linkedin_url', 'twitter_url',
                  'facebook_url', 'google_url', 'github_url', 'website_url',
                  'bio', 'email', 'is_want_news'):
        local_attr = locals()[field]
        if field == 'email':
            local_attr = local_attr.lower()
        if getattr(user, field) != local_attr:
            logger.info('Update the "{}" field from {} to {}'
                        .format(field, getattr(user, field), local_attr))
            setattr(user, field, local_attr)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        if select_user_by_email(session, user.email) is not None:
            message = 'email is already in use'

            logger.error(message)
            raise NameClashError(message)
        else:
            logger.error(repr(e))
            raise e
