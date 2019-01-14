import logging

import click

from ramputils import generate_ramp_config
from ramputils import read_config

from .utils import setup_db

from .tools import team
from .tools import user

logger = logging.getLogger('RAMP-DATABASE-CLI')


@click.group()
@click.option("--config", help='Configuration file in YAML format')
@click.pass_context
def main(ctx, config):
    ctx.obj['config_filename'] = config
    ctx.obj['config'] = read_config(config)
    ctx.obj['db'], ctx.obj['Session'] = setup_db(
        ctx.obj['config']['sqlalchemy']
    )
    logger.info('Create an engine to the database')


@main.command()
@click.option('--login', help='Login')
@click.option('--password', help='Password')
@click.option('--lastname', help="User's last name")
@click.option('--firstname', help="User's first name")
@click.option('--email', help="User's email")
@click.option('--access_level', default='user',
              help="User's administration rights")
@click.option('--hidden_notes', default='',
              help="User's additional hidden notes")
@click.pass_context
def add_user(ctx, login, password, lastname, firstname, email,
                access_level, hidden_notes):
    with ctx.obj['db'].connect() as conn:
        session = ctx.obj['Session'](bind=conn)
        new_user = user.add_user(session, login, password, lastname,
                                 firstname, email, access_level,
                                 hidden_notes)
        logger.info('Create a new user: {}'.format(new_user))


@main.command()
@click.option('--login', help="User's login to be approved")
@click.pass_context
def approve_user(ctx, login):
    ramp_config = generate_ramp_config(ctx.obj['config'])
    with ctx.obj['db'].connect() as conn:
        session = ctx.obj['Session'](bind=conn)
        user.approve_user(session, login)
        logger.info('Approved user: {}'.format(user))


@main.command()
@click.option('--name', help='Name of the team to sign up')
@click.pass_context
def sign_up_team(ctx, name):
    ramp_config = generate_ramp_config(ctx.obj['config'])
    with ctx.obj['db'].connect() as conn:
        session = ctx.obj['Session'](bind=conn)
        team.sign_up_team(session, ramp_config['event_name'], name)
        logger.info('Sign up the team "{}" to the event "{}"'
                    .format(name, ramp_config['event_name']))


def start():
    main(obj={})


if __name__ == '__main__':
    start()
