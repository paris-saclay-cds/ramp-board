import click

from ramputils import generate_ramp_config
from ramputils import deploy


@click.group()
def main():
    pass


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
def deploy_ramp_event(config):
    deploy.deploy_ramp_event(config)


def start():
    main()


if __name__ == '__main__':
    start()
