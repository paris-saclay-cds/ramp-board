import click

from ramputils import generate_ramp_config
from ramputils import deploy


@click.group()
@click.option("--config", help='Configuration file in YAML format')
@click.pass_context
def main(ctx, config):
    ctx.obj['config_filename'] = config


@main.command()
@click.pass_context
def deploy_ramp_event(ctx):
    ramp_config = generate_ramp_config(ctx.obj['config_filename'])
    deploy.deploy_ramp_event(ctx.obj['config_filename'])


def start():
    main(obj={})


if __name__ == '__main__':
    start()
