from __future__ import print_function, absolute_import, unicode_literals

import argparse

from .tools import score_submission
from .config import read_backend_config


def init_parser():
    """Defines command-line interface"""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Score a submission')
    parser.add_argument('config', type=str,
                        help='Backend configuration file with database '
                             'connexion and RAMP event details.')
    parser.add_argument('submission_id', type=int,
                        help='ID of the submission')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Increase verbosity')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()

    config = read_backend_config(args.config)
    score_submission(config, args.submission_id)


if __name__ == '__main__':
    main()
