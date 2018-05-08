from __future__ import print_function, absolute_import, unicode_literals

import sys
import logging
import argparse
from argparse import RawTextHelpFormatter

from rampbkd.aws.api import launch_ec2_instance_and_train
from rampbkd.config import read_backend_config
from rampbkd.api import get_submission_by_name

def init_parser():
    """Defines command-line interface"""
    desc = (
        'Train a submission on AWS.\nTwo ways of specifying the submission '
        'are available.\nEither we give the submission id or name.\n\n'
        'Use ramp_aws_train config.yml --id=<submission id> if you want to '
        'specify submission by name.\n\nUse ramp_aws_train config.yml '
        '--event=<event name> --team=<team name> --name=<submission name>'
        '\nif you want to specify submission by name.')
    parser = argparse.ArgumentParser(
        prog=__file__,
        description=desc,
        formatter_class=RawTextHelpFormatter)
    parser.add_argument('config', type=str,
                        help='Backend configuration file with database') 
    parser.add_argument('--id', type=int,
                        help='Submission ID')
    parser.add_argument('--event', type=str,
                        help='Event name')
    parser.add_argument('--team', type=str,
                        help='Team name')
    parser.add_argument('--name', type=str,
                        help='Submission name')
    parser.add_argument('--log-level', type=str, default='INFO',
                        help='Log level : DEBUG/INFO/WARNING/ERROR/CRITICAL')
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    logger = logging.getLogger('ramp_aws')
    logger.setLevel(args.log_level)
    config = read_backend_config(args.config)
    
    if args.id:
        submission_id = args.id
    else:
        try:
            submission = get_submission_by_name(
                config,
                args.event,
                args.team,
                args.name
            )
        except Exception as ex:
            print('Submission not found. Reasons:')
            print(ex)
            sys.exit(1)
        submission_id = submission.id
    launch_ec2_instance_and_train(config, submission_id)


if __name__ == '__main__':
    main()
