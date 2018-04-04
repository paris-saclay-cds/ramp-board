from __future__ import print_function, absolute_import, unicode_literals

import argparse

from .api import get_submissions


def init_parser():
    """Defines command-line interface"""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Retrieve the submissions with state 'new'")

    parser.add_argument('config', type=str,
                        help='Backend configuration file with database '
                             'connexion and RAMP event details.')
    parser.add_argument('event_name', type=str,
                        help='Event name of the submission.')

    return parser


def main():
    # Parse command-line arguments
    parser = init_parser()
    args = parser.parse_args()

    files = get_submissions(args.config, args.event_name, state='new')

    for f in files:
        print(f)


if __name__ == '__main__':
    main()
