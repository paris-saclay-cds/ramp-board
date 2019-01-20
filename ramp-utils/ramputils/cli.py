import click

from ramputils import deploy

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
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
