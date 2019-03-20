import click

from .wsgi import make_app

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file in YAML format')
@click.option("--port", default=8080, show_default=True,
              help='The port where to launch the website')
@click.option("--host", default='127.0.0.1', show_default=True,
              help='The IP address where to launch the website')
def test_launch(config, port, host):
    app = make_app(config)
    app.run(port=port, use_reloader=False,
            host=host, processes=1, threaded=False)


def start():
    main()


if __name__ == '__main__':
    start()
