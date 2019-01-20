import click

from ramputils import generate_ramp_config
from ramputils import deploy


@click.group()
def main():
    pass


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option("--cloning/--no-cloning", default=True,
              help='Whether or not to clone the RAMP kit and data '
              'repositories.')
@click.option('--force', is_flag=True,
              help='Whether or not to potentially overwrite the '
              'repositories, problem and event in the database.')
def deploy_ramp_event(config, cloning, force):
    deploy.deploy_ramp_event(config, cloning, force)


def start():
    main()


if __name__ == '__main__':
    start()
