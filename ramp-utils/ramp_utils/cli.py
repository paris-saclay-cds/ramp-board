import os
import shutil

import click

from ramp_utils import deploy


HERE = os.path.dirname(__file__)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Command-lines for setting and deploying RAMP server and events.

    These command-lines are used to create the configuration file for your
    RAMP server and events. In addition, it will allow you to deploy easily
    RAMP events.
    """
    pass


@main.command()
@click.option("--deployment-dir", default='.', show_default=True,
              help='The directory where to create a config file.')
@click.option('--force', is_flag=True,
              help='Whether or not to potentially overwrite the '
              'repositories, problem and event in the database.')
def init(deployment_dir, force):
    """Initialize the deployment directory with a template config.yml"""
    template = os.path.join(HERE, 'template', 'database_config_template.yml')
    destination = os.path.join(deployment_dir, 'config.yml')
    if os.path.isfile(destination) and not force:
        click.echo(
            "Config file already exists. Specify --force to overwrite it.")
        return
    shutil.copy(template, destination)
    click.echo("Created {}".format(destination))
    click.echo("You still need to modify it to fill correct parameters.")


@main.command()
@click.option("--name", help='The name of the event.', required=True)
@click.option("--deployment-dir", default='.', show_default=True,
              help='The directory where to create a config file.')
@click.option('--force', is_flag=True,
              help='Whether or not to potentially overwrite the '
              'repositories, problem and event in the database.')
def init_event(name, deployment_dir, force):
    """Initialize the event directory with a template config.yml"""

    # create directories
    events_dir = os.path.join(deployment_dir, 'events')
    if not os.path.isdir(events_dir):
        os.mkdir(events_dir)

    event_dir = os.path.join(events_dir, name)
    if os.path.isdir(event_dir):
        if force:
            shutil.rmtree(event_dir)
        else:
            click.echo(
                "{} already exists. Specify --force to overwrite it.".format(
                    event_dir))
            return
    os.mkdir(event_dir)

    # copy + edit config template
    template = os.path.join(HERE, 'template', 'ramp_config_template.yml')
    destination = os.path.join(event_dir, 'config.yml')
    with open(destination, 'w') as dest:
        with open(template, 'r') as src:
            content = src.read()
            content = content.replace('<name>', name)
            dest.write(content)

    click.echo("Created {}".format(destination))
    click.echo("You still need to modify it to fill correct parameters.")


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file in YAML format containing the database '
              'information')
@click.option("--event-config", default='config.yml', show_default=True,
              help='Configuration file in YAML format containing the RAMP '
              'information')
@click.option("--cloning/--no-cloning", default=True, show_default=True,
              help='Whether or not to clone the RAMP kit and data '
              'repositories.')
@click.option('--force', is_flag=True,
              help='Whether or not to potentially overwrite the '
              'repositories, problem and event in the database.')
def deploy_event(config, event_config, cloning, force):
    """Deploy event (add problem and event to the database, optionally clone
    kit and data)
    """
    deploy.deploy_ramp_event(config, event_config, cloning, force)


def start():
    main()


if __name__ == '__main__':
    start()
