import logging

import click

from ramputils import read_config
from ramputils import generate_worker_config

from rampbkd.dispatcher import Dispatcher
from rampbkd.dispatcher import CondaEnvWorker

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Command-line to launch engine to process RAMP submission."""
    pass


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format containing the database '
              'information.')
@click.option("--event-config", default='config.yml',
              help='Configuration file in YAML format containing the RAMP '
              'event information.')
@click.option('--worker-type', default='CondaEnvWorker',
              help='Type of worker to use')
@click.option('--n-worker', default=-1,
              help='Number of worker to start in parallel')
@click.option('--hunger-policy', default='exit',
              help='Policy to apply in case that there is no anymore workers'
              'to be processed')
@click.option('-v', '--verbose', is_flag=True)
def dispatcher(config, event_config, worker_type, n_worker, hunger_policy, verbose):
    """Launch the RAMP dispatcher.

    The RAMP dispatcher is in charge of starting RAMP workers, collecting
    results from them, and update the database.
    """
    if verbose:
        logging.basicConfig(format='%(levelname)s %(name)s %(message)s',
                            level=logging.DEBUG)
    config = read_config(config)
    event_config = read_config(event_config)
    disp = Dispatcher(
        config=config, event_config=event_config, worker=globals()[worker_type],
        n_worker=n_worker, hunger_policy=hunger_policy
    )
    disp.launch()


@main.command()
@click.option("--config", default='config.yml',
              help='Configuration file in YAML format')
@click.option('--worker-type', default='CondaEnvWorker',
              help='Type of worker to use')
@click.option('--submission', help='The submission name')
@click.option('-v', '--verbose', is_flag=True)
def worker(config, worker_type, submission, verbose):
    """Launch a standalone RAMP worker.

    The RAMP worker is in charger of processing a single submission by
    specifying the different locations (kit, data, logs, predictions)
    """
    if verbose:
        logging.basicConfig(format='%(levelname)s %(name)s %(message)s',
                            level=logging.DEBUG)
    config = read_config(config)
    worker_params = generate_worker_config(config)
    work = globals()[worker_type](worker_params, submission)
    work.launch()


def start():
    main()


if __name__ == '__main__':
    start()
