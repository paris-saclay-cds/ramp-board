from __future__ import print_function, absolute_import, unicode_literals

import argparse

from rampbkd.aws.api import train_loop
from rampbkd.config import read_backend_config


def init_parser():
    """Defines command-line interface"""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Train loop using AWS EC2 as a backend')
    parser.add_argument('config', type=str,
                        help='Backend configuration file with database '
                             'connexion and RAMP event details.')
    parser.add_argument('event_name', type=str, help='Event name')
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    config = read_backend_config(args.config)
    train_loop(config, args.event_name)


if __name__ == '__main__':
    main()
