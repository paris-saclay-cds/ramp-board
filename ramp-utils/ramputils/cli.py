import logging

import click

from ramputils import generate_ramp_config
from ramputils import read_config

from ramputils import deploy

logger = logging.getLogger('UTILS-CLI')


@click.group()
@click.option("--config", help='Configuration file in YAML format')
@click.pass_context
def main(ctx, config):
    ctx.obj['config_filename'] = config


@main.command()
@click.pass_context
def deploy_ramp_event(ctx):
    deploy.deploy_ramp_event(ctx.obj['config_filename'])
    logger.info('Deploy the RAMP event: {}'
                .format(ctx.obj['config']['ramp']['event']))


def start():
    main(obj={})


if __name__ == '__main__':
    start()
