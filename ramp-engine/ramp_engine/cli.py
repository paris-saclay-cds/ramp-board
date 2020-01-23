import logging

import click

from ramp_utils import read_config
from ramp_utils import generate_worker_config

from ramp_engine.dispatcher import Dispatcher
from ramp_engine import available_workers


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Command-line to launch engine to process RAMP submission."""
    pass


@main.command()
@click.option("--config", default='config.yml', show_default=True,
              help='Configuration file in YAML format containing the database '
              'information.')
@click.option("--event-config", show_default=True,
              help='Configuration file in YAML format containing the RAMP '
              'event information.')
@click.option('--n-workers', default=-1, show_default=True,
              help='Number of worker to start in parallel.')
@click.option('--n-threads', default=None, show_default=None, type=int,
              help='Number of threads used by each worker.')
@click.option('--hunger-policy', default='exit', show_default=True,
              help='Policy to apply in case that there is no anymore workers'
              'to be processed.')
@click.option('-v', '--verbose', count=True)
def dispatcher(config, event_config, n_workers, n_threads, hunger_policy,
               verbose):
    """Launch the RAMP dispatcher.

    The RAMP dispatcher is in charge of starting RAMP workers, collecting
    results from them, and update the database.
    """
    if verbose:
        if verbose == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=level, datefmt='%Y:%m:%d %H:%M:%S'
        )
    internal_event_config = read_config(event_config)
    worker_type = available_workers[
        internal_event_config['worker']['worker_type']
    ]
    disp = Dispatcher(
        config=config, event_config=event_config, worker=worker_type,
        n_workers=n_workers, n_threads=n_threads, hunger_policy=hunger_policy
    )
    disp.launch()


@main.command()
@click.option("--event-config", default='config.yml', show_default=True,
              help='Configuration file in YAML format containing the RAMP '
              'event information.')
@click.option('--submission', help='The submission name')
@click.option('-v', '--verbose', is_flag=True)
def worker(event_config, submission, verbose):
    """Launch a standalone RAMP worker.

    The RAMP worker is in charger of processing a single submission by
    specifying the different locations (kit, data, logs, predictions)
    """
    if verbose:
        if verbose == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            level=level, datefmt='%Y:%m:%d %H:%M:%S'
        )
    config = read_config(event_config)
    worker_params = generate_worker_config(config)
    worker_type = available_workers[worker_params['worker_type']]
    worker = worker_type(worker_params, submission)
    worker.launch()


def start():
    main()


if __name__ == '__main__':
    start()
