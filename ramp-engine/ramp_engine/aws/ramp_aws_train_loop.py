from __future__ import print_function, absolute_import, unicode_literals

import sys
import logging
import argparse

from .aws_train import train_loop

from ramp_engine.aws.api import validate_config
from ramp_engine.config import read_backend_config


def init_parser():
    """Defines command-line interface"""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Train loop using AWS EC2 as a backend')
    parser.add_argument('config', type=str,
                        help='Backend configuration file with database '
                             'connexion and RAMP event details.')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Log level : DEBUG/INFO/WARNING/ERROR/CRITICAL')
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    logger = logging.getLogger('ramp_aws')
    logger.setLevel(args.log_level)
    config = read_backend_config(args.config)
    validate_config(config)
    try:
        event_name = config['ramp']['event_name']
    except KeyError:
        print('Cannot find event_name in section ramp of the {}'
              .format(args.config))
        sys.exit(1)
    train_loop(config, event_name)


if __name__ == '__main__':
    main()
