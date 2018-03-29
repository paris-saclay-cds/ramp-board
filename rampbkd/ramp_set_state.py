from __future__ import print_function, absolute_import, unicode_literals

import sys
import argparse

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import URL

from .model import Base
from .tools import set_state
from .config import read_backend_config

SUBSTATES = [
    'new',               # submitted by user to frontend server
    'checked',           # not used, checking is part of the workflow now
    'checking_error',    # not used, checking is part of the workflow now
    'trained',           # training finished normally on the backend server
    'training_error',    # training finished abnormally on the backend server
    'validated',         # validation finished normally on the backend server
    'validating_error',  # validation finished abnormally on the backend server
    'tested',            # testing finished normally on the backend server
    'testing_error',     # testing finished abnormally on the backend server
    'training',          # training is running normally on the backend server
    'sent_to_training',  # frontend server sent submission to backend server
    'scored',            # submission scored on the frontend server.Final state
]
"""list of int: enumeration of available submission states"""


def init_parser():
    """Defines command-line interface"""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description='Change the state of a given submission on the '
                    'server database')

    parser.add_argument('config', type=str, nargs=1,
                        help='Backend configuration file with database '
                             'connexion and RAMP event details.')
    parser.add_argument('-E', '--event_name', type=str,
                        help='Even name of the submission.')
    parser.add_argument('-T', '--team_name', type=str,
                        help='Team name of the submission.')
    parser.add_argument('submission_name', type=str,
                        help='Submission name = ID.')
    parser.add_argument('state', choices=SUBSTATES,
                        help='New state of the submission.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Increase verbosity')

    return parser


def main():
    # Parse command-line arguments
    parser = init_parser()
    args = parser.parse_args()

    # Read config from external file
    try:
        conf = read_backend_config(args.config)
    except FileNotFoundError:
        print('Config file not found')
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(2)

    # Override config parameters with command-line arguments
    if args.event_name is not None:
        conf['ramp'].update({'event_name': args.event_name})

    if args.team_name is not None:
        conf['ramp'].update({'team_name': args.team_name})

    conf['ramp'].update({'submission_name': args.submission_name,
                         'state': args.state})

    # Create database url
    db_url = URL(**conf['sqlalchemy'])
    db = create_engine(db_url)

    # Create a configured "Session" class
    Session = sessionmaker(db)

    # Link the relational model to the database
    Base.metadata.create_all(db)

    # Connect to the dabase and perform action
    with db.connect() as conn:
        session = Session(bind=conn)

        set_state(**conf['ramp'])

        session.commit()

    if args.verbose:
        print("State of {} changed to {}"
              .format(args.submission_name, args.state))


if __name__ == '__main__':
    main()
