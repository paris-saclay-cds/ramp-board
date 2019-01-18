import click

from ramputils import generate_ramp_config
from ramputils import read_config

from .utils import session_scope

from .tools import team
from .tools import user


@click.group()
def main():
    pass


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--login', help='Login')
@click.option('--password', help='Password')
@click.option('--lastname', help="User's last name")
@click.option('--firstname', help="User's first name")
@click.option('--email', help="User's email")
@click.option('--access_level', default='user',
              help="User's administration rights")
@click.option('--hidden_notes', default='',
              help="User's additional hidden notes")
def add_user(config, login, password, lastname, firstname, email,
             access_level, hidden_notes):
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user.add_user(session, login, password, lastname, firstname, email,
                      access_level, hidden_notes)


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--login', help="User's login to be approved")
def approve_user(config, login):
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user.approve_user(session, login)


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--name', help='Name of the team to sign up')
def sign_up_team(config, name):
    config = read_config(config)
    ramp_config = generate_ramp_config(config)
    with session_scope(config['sqlalchemy']) as session:
        team.sign_up_team(session, ramp_config['event_name'], name)


def start():
    main()


if __name__ == '__main__':
    start()
