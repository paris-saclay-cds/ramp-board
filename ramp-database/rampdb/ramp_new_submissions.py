from __future__ import print_function, absolute_import, unicode_literals

import argparse

from ramputils import read_config

from .tools import get_submissions


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

    config = read_config(args.config, filter_section='sqlalchemy')
    # TODO: this is not the files actually
    # TODO: it is buggy
    files = get_submissions(config, args.event_name, state='new')

    for f in files:
        print(f)


if __name__ == '__main__':
    main()
