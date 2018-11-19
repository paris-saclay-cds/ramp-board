from __future__ import print_function, absolute_import, unicode_literals

import argparse

from rampbkd.aws.api import launch_ec2_instance_and_train
from rampbkd.config import read_backend_config


def init_parser():
    """Defines command-line interface"""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='train a submission on aws')
    parser.add_argument('config', type=str,
                        help='Backend configuration file with database '
                             'connexion and RAMP event details.')
    parser.add_argument('submission_id', type=int,
                        help='ID of the submission')
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    config = read_backend_config(args.config)
    launch_ec2_instance_and_train(config, args.submission_id)


if __name__ == '__main__':
    main()
