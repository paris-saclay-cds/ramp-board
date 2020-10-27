from collections import defaultdict

import click
import os
import pandas as pd
import shutil

from ramp_utils import read_config
from ramp_utils import generate_ramp_config

from .utils import session_scope

from .tools import event as event_module
from .tools import leaderboard as leaderboard_module
from .tools import submission as submission_module
from .tools import team as team_module
from .tools import user as user_module

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Command-line to interact directly with the database."""
    pass


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--login', help='Login')
@click.option('--password', help='Password')
@click.option('--lastname', help="User's last name")
@click.option('--firstname', help="User's first name")
@click.option('--email', help="User's email")
@click.option('--access-level', default='user', show_default=True,
              help="User's administration rights")
@click.option('--hidden-notes', default='', show_default=True,
              help="User's additional hidden notes")
def add_user(config, login, password, lastname, firstname, email,
             access_level, hidden_notes):
    """Add a new user in the database."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user_module.add_user(session, login, password, lastname, firstname,
                             email, access_level, hidden_notes)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--login', help="User's login to be removed")
def delete_user(config, login):
    """Delete a user which asked to sign-up to RAMP studio."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user_module.delete_user(session, login)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--login', help="User's login to be approved")
def approve_user(config, login):
    """Approve a user which asked to sign-up to RAMP studio."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user_module.approve_user(session, login)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--login', help="User's login to be made admin")
def make_user_admin(config, login):
    """Make a user a RAMP admin."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user_module.make_user_admin(session, login)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--login', help="User's login to be made admin")
@click.option('--access-level', help="The access level to grant the user."
              "One of {'asked', 'user', 'admin'}", default='user',
              show_default=True)
def set_user_access_level(config, login, access_level):
    """Change the access level of a RAMP user."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        user_module.set_user_access_level(session, login, access_level)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--event', help='Name of the event')
@click.option('--team', help='Name of the team')
def sign_up_team(config, event, team):
    """Sign-up a user (or team) for a RAMP event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        team_module.sign_up_team(session, event, team)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--event', help='Name of the event')
@click.option('--team', help='Name of the team')
def delete_event_team(config, event, team):
    """Delete a link between a user (or team) and a RAMP event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        team_module.delete_event_team(session, event, team)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option('--problem', help='Name of the problem')
@click.option('--kit-dir', help='Path to the RAMP kit')
@click.option('--data-dir', help='Path to the RAMP data')
@click.option('--force', default=False, show_default=True,
              help='Whether or not to overwrite the problem if it exists')
def add_problem(config, problem, kit_dir, data_dir, force):
    """Add a RAMP problem in the database."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        event_module.add_problem(session, problem, kit_dir, data_dir, force)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--problem", help='Name of the problem')
@click.option("--event", help='Name of the event')
@click.option("--title", help='Title of the event')
@click.option("--sandbox", default='starting-kit', help='Name of the sandbox')
@click.option('--submissions-dir',
              help='Path to the deployment RAMP submissions path.')
@click.option('--is-public', default=False, show_default=True,
              help='Whether or not the event should be public')
@click.option('--force', default=False, show_default=True,
              help='Whether or not to overwrite the problem if it exists')
def add_event(config, problem, event, title, sandbox, submissions_dir,
              is_public, force):
    """Add an event in the database."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        event_module.add_event(session, problem, event, title, sandbox,
                               submissions_dir, is_public, force)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--event", help='Name of the event')
@click.option("--user", help='Name of the user becoming event admin')
def add_event_admin(config, event, user):
    """Make a user admin of a specific RAMP event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        event_module.add_event_admin(session, event, user)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--event", help='Name of the event')
@click.option("--team", help='Name of the team')
@click.option("--submission", help='Name of the submission')
@click.option("--path", help='Path to the submission')
def add_submission(config, event, team, submission, path):
    """Add a submission to the database."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        submission_module.add_submission(session, event, team, submission,
                                         path)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--config-event", required=True,
              help='Path to configuration file YAML format '
              'containing the database information, eg config.yml')
@click.option('--dry-run', is_flag=True,
              help='Emulate the removal without taking action. Basically, '
              'only the printing information will be shown. The deletion will '
              'not be done.')
@click.option('--from-disk', is_flag=True,
              help='Flag to remove the event folder from the disk as well.')
@click.option('--force', is_flag=True,
              help='Flag to force a removal, even from the disk, when an '
              'event is not in the database.')
def delete_event(config, config_event, dry_run, from_disk, force):
    """Delete event."""
    internal_config = read_config(config)
    ramp_config = generate_ramp_config(config_event, config)
    event_name = ramp_config["event_name"]

    with session_scope(internal_config['sqlalchemy']) as session:
        db_event = event_module.get_event(session, event_name)

        if db_event:
            if not dry_run:
                event_module.delete_event(session, event_name)
            click.echo(
                '{} was removed from the database'
                .format(event_name)
            )
        if from_disk:
            if not db_event and not force:
                err_msg = ('{} event not found in the database. If you want '
                           'to force removing event files from the disk, add '
                           'the option "--force".'
                           .format(event_name))
                raise click.ClickException(err_msg)
            for key in ("ramp_submissions_dir", "ramp_predictions_dir",
                        "ramp_logs_dir"):
                dir_to_remove = ramp_config[key]
                if os.path.exists(dir_to_remove):
                    if not dry_run:
                        shutil.rmtree(dir_to_remove)
                    click.echo("Removed directory:\n{}".format(dir_to_remove))
                else:
                    click.echo(
                        "Directory not found. Skip removal for the "
                        "directory:\n{}".format(dir_to_remove)
                    )
            event_dir = os.path.dirname(config_event)
            if not dry_run:
                shutil.rmtree(event_dir)
            click.echo("Removed directory:\n{}".format(event_dir))


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--config-event", required=True,
              help='Path to configuration file YAML format '
              'containing the database information, eg config.yml')
@click.option('--force', is_flag=True,
              help='Flag to force a removal of the predictions from the disk, '
                   'when an event is not in the database.')
def delete_predictions(config, config_event, force):
    """Delete event predictions from the disk."""
    internal_config = read_config(config)
    ramp_config = generate_ramp_config(config_event, config)
    event_name = ramp_config["event_name"]

    with session_scope(internal_config['sqlalchemy']) as session:
        db_event = event_module.get_event(session, event_name)

        if not db_event and not force:
            err_msg = ('{} event not found in the database. If you want '
                       'to force removing perdiction files from the disk use '
                       'the option "--force".'
                       .format(event_name))
            raise click.ClickException(err_msg)

        dir_to_remove = ramp_config["ramp_predictions_dir"]
        if os.path.exists(dir_to_remove):
            shutil.rmtree(dir_to_remove)
            click.echo("Removed directory:\n{}".format(dir_to_remove))
        else:
            err_msg = ("Directory not found:\n{}".format(dir_to_remove))
            raise click.ClickException(err_msg)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--event", help='Name of the event')
@click.option("--state", help='The state of the submissions to display')
def get_submissions_by_state(config, event, state):
    """Display the list of submission for an event in a particular state."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        submissions = submission_module.get_submissions(session, event, state)
        if submissions:
            data = defaultdict(list)
            for sub_info in submissions:
                sub = submission_module.get_submission_by_id(
                    session, sub_info[0]
                )
                data['ID'].append(sub.id)
                data['name'].append(sub.name)
                data['team'].append(sub.team)
                data['path'].append(sub.path)
                data['state'].append(sub.state)
            click.echo(pd.DataFrame(data).set_index('ID'))
        else:
            click.echo('No submission for this event and this state')


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--submission-id", help='The submission ID')
@click.option("--state", help='The state to affect to the submission')
def set_submission_state(config, submission_id, state):
    """Set the state of a submission."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        submission_module.set_submission_state(session, submission_id, state)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--event", help='The event name')
def update_leaderboards(config, event):
    """Update the leaderboards for a given event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        leaderboard_module.update_leaderboards(session, event)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--event", help='The event name')
@click.option("--user", help='The user name')
def update_user_leaderboards(config, event, user):
    """Update the user leaderboards for a given event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        leaderboard_module.update_user_leaderboards(session, event, user)


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file YAML format containing the database '
              'information')
@click.option("--event", help='The event name')
def update_all_users_leaderboards(config, event):
    """Update the leaderboards of all users for a given event."""
    config = read_config(config)
    with session_scope(config['sqlalchemy']) as session:
        leaderboard_module.update_all_user_leaderboards(session, event)


def start():
    main()


if __name__ == '__main__':
    start()
