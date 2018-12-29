from __future__ import print_function, absolute_import, unicode_literals

import argparse

from ramputils import read_config

from .tools import set_submission_state
from .model.submission import submission_states

STATES = submission_states.enums


def init_parser():
    """Defines command-line interface"""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Change the state of a given submission on the '
                    'server database')

    parser.add_argument('config', type=str,
                        help='Backend configuration file with database '
                             'connexion and RAMP event details.')
    parser.add_argument('submission_id', type=int,
                        help='ID of the submission')
    parser.add_argument('state', choices=STATES,
                        help='New state of the submission.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Increase verbosity')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    config = read_config(args.config, filter_section='sqlalchemy')
    res = set_submission_state(config, args.submission_id, args.state)

    if args.verbose and res:
        print("State of {} changed to {}"
              .format(args.submission_id, args.state))


if __name__ == '__main__':
    main()
