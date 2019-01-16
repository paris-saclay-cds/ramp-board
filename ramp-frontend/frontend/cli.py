import click

from ramputils import generate_flask_config
from ramputils import read_config

from . import create_app


@click.group()
@click.option("--config", help='Configuration file in YAML format')
@click.pass_context
def main(ctx, config):
    ctx.obj['config_filename'] = config


@main.command()
@click.option("--port", default=8080,
              help='The port where to launch the website')
@click.option("--host", default='127.0.0.1',
              help='The IP address where to launch the website')
@click.pass_context
def launch(ctx, port, host):
    config = read_config(ctx.obj['config_filename'])
    flask_config = generate_flask_config(config)
    app = create_app(flask_config)
    app.run(debug=False, port=port, use_reloader=False,
            host=host, processes=1000, threaded=False)


def start():
    main(obj={})


if __name__ == '__main__':
    start()
