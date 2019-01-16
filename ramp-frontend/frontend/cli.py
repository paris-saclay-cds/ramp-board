import click

from ramputils import generate_flask_config
from ramputils import read_config

from . import create_app


@click.group()
def main():
    pass


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
def launch(config):
    config = read_config(config)
    flask_config = generate_flask_config(config)
    app = create_app(flask_config)
    app.run(use_reloader=False, processes=1000, threaded=False)


def start():
    main()


if __name__ == '__main__':
    start()
