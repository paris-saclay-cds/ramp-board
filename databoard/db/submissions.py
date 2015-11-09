import os
import hashlib
import datetime
import pandas as pd
from collections import OrderedDict
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Enum, \
    DateTime, Boolean, UniqueConstraint
from databoard.config import submissions_path

from databoard.db.model_base import DBBase
import databoard.db.model_base as model_base
from databoard.db.teams import Team
from databoard.config import get_session, get_engine
# so set engine (call config.set_engine_and_session) before importing model
engine = get_engine()
session = get_session()


class Submission(DBBase):
    __tablename__ = 'submissions'

    submission_id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    name = Column(String, nullable=False)
    file_list = Column(String, nullable=False)
    submission_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    training_timestamp = Column(DateTime)
    scoring_timestamp = Column(DateTime)
    valid_score = Column(Float, default=0.0)  # cv
    test_score = Column(Float, default=0.0)  # holdout
    contributivity = Column(Integer, default=0)
    train_time = Column(Integer, default=0)
    test_time = Column(Integer, default=0)
    trained_state = Column(Enum('new', 'checked', 'trained', 'error', 'scored',
                                'ignore'), default='new')
    tested_state = Column(Enum(
        'new', 'tested', 'scored', 'error'), default='new')
    is_valid = Column(Boolean, default=True)  # user can delete but we keep
    is_to_ensemble = Column(Boolean, default=True)  # we can forget bad models
    notes = Column(String, default='')  # eg, why is it disqualified
    team = relationship('Team', back_populates='submissions')  # one-to-many

    UniqueConstraint(team_id, name)  # later also ramp_id

    @hybrid_property
    def is_public_leaderboard(self):
        return self.is_valid and self.trained_state == 'scored'

    @hybrid_property
    def is_private_leaderboard(self):
        return self.is_valid and self.tested_state == 'scored'

    def _get_submission_hash(self):
        sha_hasher = hashlib.sha1()
        sha_hasher.update(self.team.name)
        sha_hasher.update(self.name)
        model_hash = 'm{}'.format(sha_hasher.hexdigest())
        return model_hash

    def get_submission_path(self, submissions_path=submissions_path):
        submission_hash = self._get_submission_hash()
        team_path = os.path.join(submissions_path, self.team.name)
        submission_path = os.path.join(team_path, submission_hash)
        return team_path, submission_path

    def __repr__(self):
        repr = 'Submission(team_name={}, name={}, file_list={}, '\
            'trained_state={}, tested_state={})'.format(
                self.team.name, self.name, self.file_list, self.trained_state,
                self.tested_state)
        return repr


class DuplicateSubmissionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def make_submission(team_name, name, file_list):
    team = session.query(Team).filter_by(name=team_name).one()
    submission = session.query(Submission).filter_by(
        name=name, team=team).one_or_none()
    if submission is None:
        submission = Submission(name=name, team=team, file_list=file_list)
        session.add(submission)
    else:
        if submission.trained_state == 'new' or\
                submission.trained_state == 'error' or\
                submission.tested_state == 'error':
            submission.trained_state = 'new'
            submission.tested_state = 'new'
        else:
            raise DuplicateSubmissionError(
                'Submission "{}" of team "{}" exists already'.format(
                    name, team_name))

    # We should copy files here
    session.commit()
    return submission


def get_public_leaderboard():
    """
    Returns
    -------
    lederboard : html string
    """

    table_setup = OrderedDict([
        ('team', Team.name),
        ('submission', Submission.name),
        ('score', Submission.valid_score),
        ('contributivity', Submission.contributivity),
        ('train time', Submission.train_time),
        ('test time', Submission.test_time),
        ('submitted at', Submission.submission_timestamp),
    ])
    table_header = table_setup.keys()
    table_columns = table_setup.values()
    join = session.query(Submission, Team, *table_columns).filter(
        Team.team_id == Submission.team_id)
    submissions = join.filter(Submission.is_public_leaderboard).all()
    # We transpose, get rid of Submission and Team, then retranspose
    df = pd.DataFrame(zip(*zip(*submissions)[2:]), columns=table_header)
    df['submitted at'] = df['submitted at'].apply(
        lambda x: model_base.date_time_format(x))

    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )

    return df.to_html(**html_params)


def print_submissions():
    print('***************** List of submissions ****************')
    for submission in session.query(Submission).order_by(
            Submission.submission_id):
        print submission
