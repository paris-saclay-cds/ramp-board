from __future__ import print_function, absolute_import, unicode_literals

import sys
import logging
import argparse
from argparse import RawTextHelpFormatter

from rampbkd.aws.api import launch_ec2_instance_and_train
from rampbkd.aws.api import train_on_existing_ec2_instance
from rampbkd.config import read_backend_config
from rampbkd.api import get_submission_by_name

desc = """
Train a submission on AWS.
Two ways of specifying the submission are available.
Either we give the submission id or name.

Use ramp_aws_train config.yml --id=<submission id> if you want to 
specify submission by id.

Use ramp_aws_train config.yml --event=<event name> --team=<team name> --name=<submission name> 
if you want to specify submission by name.

By default a new ec2 instance will be created then training will be done there, 
then the instance will be killed after training.

If you want to train on an existing instance just add the option
--instance-id. Example:

ramp_aws_train config.yml --event=<event name> --team=<team name> --name=<submission name>  --instance-id=<instance id>

To find the instance id, you have to check the AWS EC2 console
or use the cli `aws` provided by amazon.

"""

def init_parser():
    """Defines command-line interface"""
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
    parser.add_argument('--instance-id', type=str,
                        help='Instance id')
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
    
    if args.instance_id:
        train_on_existing_ec2_instance(config, args.instance_id, submission_id)
    else:
        launch_ec2_instance_and_train(config, submission_id)


if __name__ == '__main__':
    main()
