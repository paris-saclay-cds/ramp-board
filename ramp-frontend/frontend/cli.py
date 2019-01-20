import click

from ramputils import generate_flask_config
from ramputils import read_config

from . import create_app

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option("--port", default=8080,
              help='The port where to launch the website')
@click.option("--host", default='127.0.0.1',
              help='The IP address where to launch the website')
def launch(config, port, host):
    config = read_config(config)
    flask_config = generate_flask_config(config)
    app = create_app(flask_config)
    app.run(port=port, use_reloader=False,
            host=host, processes=1000, threaded=False)


def start():
    main()


if __name__ == '__main__':
    start()
