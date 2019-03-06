import os
import shutil

import click

from ramp_utils import deploy


HERE = os.path.dirname(__file__)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@click.option("--deployment-dir", default='.', show_default=True,
              help='The directory where to create a config file.')
@click.option('--force', is_flag=True,
              help='Whether or not to potentially overwrite the '
              'repositories, problem and event in the database.')
def init(deployment_dir, force):
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
def deploy_ramp_event(config, event_config, cloning, force):
    deploy.deploy_ramp_event(config, event_config, cloning, force)


def start():
    main()


if __name__ == '__main__':
    start()
