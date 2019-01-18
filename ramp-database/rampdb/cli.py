import click

from ramputils import read_config

from .utils import session_scope

from .tools import event as event_module
from .tools import team as team_module
from .tools import user as user_module


@click.group()
def main():
    """Command-line to interact directly with the database."""
    pass


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--login', help='Login')
@click.option('--password', help='Password')
@click.option('--lastname', help="User's last name")
@click.option('--firstname', help="User's first name")
@click.option('--email', help="User's email")
@click.option('--access-level', default='user',
              help="User's administration rights")
@click.option('--hidden-notes', default='',
              help="User's additional hidden notes")
def add_user(config, login, password, lastname, firstname, email,
             access_level, hidden_notes):
    """Add a new user in the database."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user_module.add_user(session, login, password, lastname, firstname,
                             email, access_level, hidden_notes)


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--login', help="User's login to be approved")
def approve_user(config, login):
    """Approve a user which asked to sign-up to RAMP studio."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user_module.approve_user(session, login)


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--event', help='Name of the event')
@click.option('--team', help='Name of the team')
def sign_up_team(config, event, team):
    """Sign-up a user (or team) for a RAMP event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        team_module.sign_up_team(session, event, team)


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--problem', help='Name of the problem')
@click.option('--kits-dir', help='Path to the RAMP kits')
@click.option('--data-dir', help='Path to the RAMP data')
@click.option('--force', default=False,
              help='Whether or not to overwrite the problem if it exists')
def add_problem(config, problem, kits_dir, data_dir, force):
    """Add a RAMP problem in the database."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        event_module.add_problem(session, problem, kits_dir, data_dir, force)


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option("--problem", help='Name of the problem')
@click.option("--event", help='Name of the event')
@click.option("--title", help='Title of the event')
@click.option("--sandbox", default='starting-kit', help='Name of the sandbox')
@click.option('--submissions-dir',
              help='Path to the deployment RAMP submissions path.')
@click.option('--is-public', default=False,
              help='Whether or not the event should be public')
@click.option('--force', default=False,
              help='Whether or not to overwrite the problem if it exists')
def add_event(config, problem, event, title, sandbox, submissions_dir,
              is_public, force):
    """Add an event in the database."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        event_module.add_event(session, problem, event, title, sandbox,
                               submissions_dir, is_public, force)


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option("--event", help='Name of the event')
@click.option("--user", help='Name of the user becoming event admin')
def add_event_admin(config, event, user):
    """Make a user admin of a specific RAMP event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        event_module.add_event_admin(session, event, user)


def start():
    main()


if __name__ == '__main__':
    start()
