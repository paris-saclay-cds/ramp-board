import json
import os
import shutil
import subprocess

import click
import pandas as pd

from ramp_utils import deploy
from ramp_utils import read_config
from ramp_utils import generate_ramp_config


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


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file in YAML format containing the database '
              'information')
@click.option("--event-config", default='config.yml', show_default=True,
              help='Configuration file in YAML format containing the RAMP '
              'information')
def create_conda_env(config, event_config):
    """Create the conda environment for a specific event"""
    conda_env_name = read_config(event_config)["worker"]["conda_env"]
    ramp_config = generate_ramp_config(event_config, database_config=config)
    path_environment_file = os.path.join(
        ramp_config["ramp_kit_dir"], "environment.yml"
    )
    subprocess.run(
        ["conda", "create", "--name", conda_env_name, "--yes"]
    )
    subprocess.run(
        ["conda", "env", "update",
         "--name", conda_env_name,
         "--file", path_environment_file]
    )


@main.command()
@click.option("--event-config", default='config.yml', show_default=True,
              help='Configuration file in YAML format containing the RAMP '
              'information')
def update_conda_env(event_config):
    """Update the conda environment for a specific event"""
    conda_env = read_config(event_config, filter_section="worker")["conda_env"]
    # get the path to the right conda environment
    proc = subprocess.Popen(
        ["conda", "info", "--envs", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if stderr:
        raise ValueError(stderr.decode("utf-8"))
    conda_info = json.loads(stdout)

    if conda_env == 'base':
        python_bin_path = os.path.join(conda_info['envs'][0], 'bin')
    else:
        envs_path = conda_info['envs'][1:]
        if not envs_path:
            raise ValueError('Only the conda base environment exist. You '
                             'need to create the "{}" conda environment '
                             'to use it.'.format(conda_env))
        is_env_found = False
        for env in envs_path:
            if conda_env == os.path.split(env)[-1]:
                is_env_found = True
                python_bin_path = os.path.join(env, 'bin')
                break
        if not is_env_found:
            raise ValueError('The specified conda environment {} does not '
                             'exist. You need to create it.'
                             .format(conda_env))

    # update the conda packages
    subprocess.run(["conda", "update", "--name", conda_env, "--all", "--yes"])

    # filter the packages installed with pip
    proc = subprocess.Popen(
        ["conda", "list", "--name", conda_env, "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if stderr:
        raise ValueError(stderr.decode("utf-8"))
    packages = json.loads(stdout)

    df = pd.DataFrame(packages)
    df = df[df["channel"] == "pypi"]
    pip_packages = df["name"].tolist()

    # update the pip packages
    subprocess.run(
        [os.path.join(python_bin_path, 'pip'), "install", "-U"] + pip_packages
    )


def start():
    main()


if __name__ == '__main__':
    start()
