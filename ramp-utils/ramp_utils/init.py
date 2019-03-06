import os
import shutil

import click


HERE = os.path.dirname(__file__)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])



@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--deployment-dir", default='.', show_default=True,
              help='The directory where to create a config file.')
@click.option('--force', is_flag=True,
              help='Whether or not to potentially overwrite the '
              'repositories, problem and event in the database.')
def main(deployment_dir, force):
    template = os.path.join(HERE, 'template', 'database_config_template.yml')
    destination = os.path.join(deployment_dir, 'config.yml')
    if os.path.isfile(destination) and not force:
        click.echo(
            "Config file already exists. Specify --force to overwrite it.")
        return
    shutil.copy(template, destination)
    click.echo("Created {}".format(destination))
    click.echo("You still need to modify it to fill in )


if __name__ == '__main__':
    main()
