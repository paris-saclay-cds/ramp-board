import os
import hashlib
import datetime

import numpy as np
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import UniqueConstraint
from sqlalchemy import inspect
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from .base import Model
from .base import encode_string
from .base import get_deployment_path
from .event import EventScoreType
from .datatype import NumpyType

__all__ = [
    'Submission',
    'SubmissionScore',
    'SubmissionFile',
    'SubmissionFileType',
    'SubmissionFileTypeExtension',
    'Extension',
    'SubmissionScoreOnCVFold',
    'SubmissionOnCVFold',
    'DetachedSubmissionOnCVFold',
    'SubmissionSimilarity',
]

# evaluate right after train/test, so no need for 'scored' states
submission_states = Enum(
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
    name='submission_states')

submission_types = Enum('live', 'test', name='submission_types')


class Submission(Model):
    """An abstract (untrained) submission."""

    __tablename__ = 'submissions'

    id = Column(Integer, primary_key=True)

    event_team_id = Column(
        Integer, ForeignKey('event_teams.id'), nullable=False)
    event_team = relationship('EventTeam', backref=backref(
        'submissions', cascade='all, delete-orphan'))

    name = Column(String(20, convert_unicode=True), nullable=False)
    hash_ = Column(String, nullable=False, index=True, unique=True)
    submission_timestamp = Column(DateTime, nullable=False)
    sent_to_training_timestamp = Column(DateTime)
    training_timestamp = Column(DateTime)  # end of training

    contributivity = Column(Float, default=0.0)
    historical_contributivity = Column(Float, default=0.0)

    type = Column(submission_types, default='live')
    state = Column(String, default='new')
    # TODO: hide absolute path in error
    error_msg = Column(String, default='')
    # user can delete but we keep
    is_valid = Column(Boolean, default=True)
    # We can forget bad models.
    # If false, don't combine and set contributivity to zero
    is_to_ensemble = Column(Boolean, default=True)
    # in competitive events participants can select the submission
    # with which they want to participate in the competition
    is_in_competition = Column(Boolean, default=True)

    notes = Column(String, default='')  # eg, why is it disqualified

    train_time_cv_mean = Column(Float, default=0.0)
    valid_time_cv_mean = Column(Float, default=0.0)
    test_time_cv_mean = Column(Float, default=0.0)
    train_time_cv_std = Column(Float, default=0.0)
    valid_time_cv_std = Column(Float, default=0.0)
    test_time_cv_std = Column(Float, default=0.0)
    # the maximum memory size used when training/testing, in MB
    max_ram = Column(Float, default=0.0)
    # later also ramp_id
    UniqueConstraint(event_team_id, name, name='ts_constraint')

    def __init__(self, name, event_team):
        self.name = name
        self.event_team = event_team
        self.session = inspect(event_team).session
        sha_hasher = hashlib.sha1()
        sha_hasher.update(encode_string(self.event.name))
        sha_hasher.update(encode_string(self.team.name))
        sha_hasher.update(encode_string(self.name))
        # We considered using the id, but then it will be given away in the
        # url which is maybe not a good idea.
        self.hash_ = '{}'.format(sha_hasher.hexdigest())
        self.submission_timestamp = datetime.datetime.utcnow()
        event_score_types = EventScoreType.query.filter_by(
            event=event_team.event)
        for event_score_type in event_score_types:
            submission_score = SubmissionScore(
                submission=self, event_score_type=event_score_type)
            self.session.add(submission_score)
        self.reset()

    def __str__(self):
        return 'Submission({}/{}/{})'.format(
            self.event.name, self.team.name, self.name)

    def __repr__(self):
        repr = '''Submission(event_name={}, team_name={}, name={}, files={},
                  state={}, train_time={})'''.format(
            encode_string(self.event.name),
            encode_string(self.team.name),
            encode_string(self.name),
            self.files,
            self.state,
            self.train_time_cv_mean)
        return repr

    @hybrid_property
    def team(self):
        return self.event_team.team

    @hybrid_property
    def event(self):
        return self.event_team.event

    @property
    def official_score_function(self):
        return self.event.official_score_function

    @property
    def official_score_name(self):
        return self.event.official_score_name

    @property
    def official_score(self):
        score_dict = {score.score_name: score for score in self.scores}
        return score_dict[self.official_score_name]

    @property
    def score_types(self):
        return self.event.score_types

    @property
    def Predictions(self):
        return self.event.Predictions

    @hybrid_property
    def is_not_sandbox(self):
        return self.name != os.getenv('RAMP_SANDBOX_DIR', 'starting_kit')

    @hybrid_property
    def is_error(self):
        return (self.state == 'training_error') |\
            (self.state == 'checking_error') |\
            (self.state == 'validating_error') |\
            (self.state == 'testing_error')

    @hybrid_property
    def is_public_leaderboard(self):
        return self.is_not_sandbox & self.is_valid & (self.state == 'scored')

    @hybrid_property
    def is_private_leaderboard(self):
        return self.is_not_sandbox & self.is_valid & (self.state == 'scored')

    @property
    def path(self):
        return os.path.join(
            get_deployment_path(),
            'submissions',
            'submission_' + '{0:09d}'.format(self.id))

    @property
    def module(self):
        return self.path.lstrip('./').replace('/', '.')

    @property
    def f_names(self):
        return [file.f_name for file in self.files]

    @property
    def link(self):
        return self.files[0].link

    @property
    def full_name_with_link(self):
        return '<a href={}>{}/{}/{}</a>'.format(
            self.link, self.event.name, self.team.name, self.name[:20])

    @property
    def name_with_link(self):
        return '<a href={}>{}</a>'.format(self.link, self.name[:20])

    @property
    def state_with_link(self):
        return '<a href=/{}>{}</a>'.format(
            os.path.join(self.hash_, 'error.txt'), self.state)

    def ordered_scores(self, score_names):
        """Iterator yielding SubmissionScores.

        Ordered according to score_names. Called by get_public_leaderboard
        and get_private_leaderboard, making sure scores are listed in the
        correct column.

        Parameters
        ----------
        score_names : list of strings

        Return
        ----------
        scores : iterator of SubmissionScore objects
        """
        score_dict = {score.score_name: score for score in self.scores}
        for score_name in score_names:
            yield score_dict[score_name]

    # These were constructing means and stds by fetching fold times. It was
    # slow because submission_on_folds contain also possibly large predictions
    # If postgres solves this issue (which can be tested on the mean and std
    # scores on the private leaderbord), the corresponding columns (which are
    # now redundant) can be deleted and these can be uncommented.
    # @property
    # def train_time_cv_mean(self):
    #     return np.mean([ts.train_time for ts in self.on_cv_folds])

    # @property
    # def valid_time_cv_mean(self):
    #     return np.mean([ts.valid_time for ts in self.on_cv_folds])

    # @property
    # def test_time_cv_mean(self):
    #     return np.mean([ts.test_time for ts in self.on_cv_folds])

    # @property
    # def train_time_cv_std(self):
    #     return np.std([ts.train_time for ts in self.on_cv_folds])

    # @property
    # def valid_time_cv_std(self):
    #     return np.std([ts.valid_time for ts in self.on_cv_folds])

    # @property
    # def test_time_cv_std(self):
    #     return np.std([ts.test_time for ts in self.on_cv_folds])

    def set_state(self, state):
        self.state = state
        for submission_on_cv_fold in self.on_cv_folds:
            submission_on_cv_fold.state = state

    def reset(self):
        self.contributivity = 0.0
        self.state = 'new'
        self.error_msg = ''
        for score in self.scores:
            score.valid_score_cv_bag = score.event_score_type.worst
            score.test_score_cv_bag = score.event_score_type.worst
            score.valid_score_cv_bags = None
            score.test_score_cv_bags = None

    def set_error(self, error, error_msg):
        self.reset()
        self.state = error
        self.error_msg = error_msg
        for submission_on_cv_fold in self.on_cv_folds:
            submission_on_cv_fold.set_error(error, error_msg)

    # contributivity could be a property but then we could not query on it
    def set_contributivity(self):
        self.contributivity = 0.0
        if self.is_public_leaderboard:
            # we share a unit of 1. among folds
            unit_contributivity = 1. / len(self.on_cv_folds)
            for submission_on_cv_fold in self.on_cv_folds:
                self.contributivity +=\
                    unit_contributivity * submission_on_cv_fold.contributivity

    def set_state_after_training(self):
        self.training_timestamp = datetime.datetime.utcnow()
        states = [submission_on_cv_fold.state
                  for submission_on_cv_fold in self.on_cv_folds]
        if all(state in ['tested'] for state in states):
            self.state = 'tested'
        elif all(state in ['tested', 'validated'] for state in states):
            self.state = 'validated'
        elif all(state in ['tested', 'validated', 'trained']
                 for state in states):
            self.state = 'trained'
        elif any(state == 'training_error' for state in states):
            self.state = 'training_error'
            i = states.index('training_error')
            self.error_msg = self.on_cv_folds[i].error_msg
        elif any(state == 'validating_error' for state in states):
            self.state = 'validating_error'
            i = states.index('validating_error')
            self.error_msg = self.on_cv_folds[i].error_msg
        elif any(state == 'testing_error' for state in states):
            self.state = 'testing_error'
            i = states.index('testing_error')
            self.error_msg = self.on_cv_folds[i].error_msg
        if 'error' not in self.state:
            self.error_msg = ''


class SubmissionScore(Model):
    __tablename__ = 'submission_scores'

    id = Column(Integer, primary_key=True)
    submission_id = Column(
        Integer, ForeignKey('submissions.id'), nullable=False)
    submission = relationship('Submission', backref=backref(
        'scores', cascade='all, delete-orphan'))

    event_score_type_id = Column(
        Integer, ForeignKey('event_score_types.id'), nullable=False)
    event_score_type = relationship(
        'EventScoreType', backref=backref('submissions'))

    # These are cv-bagged scores. Individual scores are found in
    # SubmissionToTrain
    valid_score_cv_bag = Column(Float)  # cv
    test_score_cv_bag = Column(Float)  # holdout
    # we store the partial scores so to see the saturation and
    # overfitting as the number of cv folds grow
    valid_score_cv_bags = Column(NumpyType)
    test_score_cv_bags = Column(NumpyType)

    @property
    def score_name(self):
        return self.event_score_type.name

    @property
    def score_function(self):
        return self.event_score_type.score_function

    # default display precision in n_digits
    @property
    def precision(self):
        return self.event_score_type.precision

    @property
    def train_score_cv_mean(self):
        return np.mean([ts.train_score for ts in self.on_cv_folds])

    @property
    def valid_score_cv_mean(self):
        return np.mean([ts.valid_score for ts in self.on_cv_folds])

    @property
    def test_score_cv_mean(self):
        return np.mean([ts.test_score for ts in self.on_cv_folds])

    @property
    def train_score_cv_std(self):
        return np.std([ts.train_score for ts in self.on_cv_folds])

    @property
    def valid_score_cv_std(self):
        return np.std([ts.valid_score for ts in self.on_cv_folds])

    @property
    def test_score_cv_std(self):
        return np.std([ts.test_score for ts in self.on_cv_folds])


# TODO: we should have a SubmissionWorkflowElementType table, describing the
# type of files we are expecting for a given RAMP. Fast unit test should be
# set up there, and each file should be unit tested right after submission.
# Kozmetics: erhaps mark which file the leaderboard link should point to (right
# now it is set to the first file in the list which is arbitrary).
# We will also have to handle auxiliary files (like csvs or other classes).
# User interface could have a sinlge submission form with a menu containing
# the file names for a given ramp + an "other" field when users will have to
# name their files
class SubmissionFile(Model):
    __tablename__ = 'submission_files'

    id = Column(Integer, primary_key=True)
    submission_id = Column(
        Integer, ForeignKey('submissions.id'), nullable=False)
    submission = relationship(
        'Submission',
        backref=backref('files', cascade='all, delete-orphan'))

    # e.g. 'regression', 'external_data'
    workflow_element_id = Column(
        Integer, ForeignKey('workflow_elements.id'),
        nullable=False)
    workflow_element = relationship(
        'WorkflowElement', backref=backref('submission_files'))

    # e.g., ('code', 'py'), ('data', 'csv')
    submission_file_type_extension_id = Column(
        Integer, ForeignKey('submission_file_type_extensions.id'),
        nullable=False)
    submission_file_type_extension = relationship(
        'SubmissionFileTypeExtension', backref=backref('submission_files'))

    # eg, 'py'
    @property
    def is_editable(self):
        return self.workflow_element.is_editable

    # eg, 'py'
    @property
    def extension(self):
        return self.submission_file_type_extension.extension.name

    # eg, 'regressor'
    @property
    def type(self):
        return self.workflow_element.type

    # eg, 'regressor', Normally same as type, except when type appears more
    # than once in workflow
    @property
    def name(self):
        return self.workflow_element.name

    # Complete file name, eg, 'regressor.py'
    @property
    def f_name(self):
        return self.type + '.' + self.extension

    @property
    def link(self):
        return '/' + os.path.join(self.submission.hash_, self.f_name)

    @property
    def path(self):
        return os.path.join(self.submission.path, self.f_name)

    @property
    def name_with_link(self):
        return '<a href="' + self.link + '">' + self.name + '</a>'

    def get_code(self):
        with open(self.path) as f:
            code = f.read()
        return code

    def set_code(self, code):
        code.encode('ascii')  # to raise an exception if code is not ascii
        with open(self.path, 'w') as f:
            f.write(code)

    def __repr__(self):
        return 'SubmissionFile(name={}, type={}, extension={}, path={})'.\
            format(self.name, self.type, self.extension, self.path)


class SubmissionFileTypeExtension(Model):
    __tablename__ = 'submission_file_type_extensions'

    id = Column(Integer, primary_key=True)

    type_id = Column(
        Integer, ForeignKey('submission_file_types.id'), nullable=False)
    type = relationship(
        'SubmissionFileType', backref=backref('extensions'))

    extension_id = Column(
        Integer, ForeignKey('extensions.id'), nullable=False)
    extension = relationship(
        'Extension', backref=backref('submission_file_types'))

    UniqueConstraint(type_id, extension_id, name='we_constraint')

    @property
    def file_type(self):
        return self.type.name

    @property
    def extension_name(self):
        return self.extension.name


class SubmissionFileType(Model):
    __tablename__ = 'submission_file_types'

    id = Column(Integer, primary_key=True)
    # eg. 'code', 'text', 'data'
    name = Column(String, nullable=False, unique=True)
    is_editable = Column(Boolean, default=True)
    max_size = Column(Integer, default=None)


class Extension(Model):
    __tablename__ = 'extensions'

    id = Column(Integer, primary_key=True)
    # eg. 'py', 'csv', 'R'
    name = Column(String, nullable=False, unique=True)


class SubmissionScoreOnCVFold(Model):
    __tablename__ = 'submission_score_on_cv_folds'

    id = Column(Integer, primary_key=True)
    submission_on_cv_fold_id = Column(
        Integer, ForeignKey('submission_on_cv_folds.id'), nullable=False)
    submission_on_cv_fold = relationship(
        'SubmissionOnCVFold', backref=backref(
            'scores', cascade='all, delete-orphan'))

    submission_score_id = Column(
        Integer, ForeignKey('submission_scores.id'), nullable=False)
    submission_score = relationship('SubmissionScore', backref=backref(
        'on_cv_folds', cascade='all, delete-orphan'))

    train_score = Column(Float)
    valid_score = Column(Float)
    test_score = Column(Float)

    UniqueConstraint(
        submission_on_cv_fold_id, submission_score_id, name='ss_constraint')

    @property
    def name(self):
        return self.event_score_type.name

    @property
    def event_score_type(self):
        return self.submission_score.event_score_type

    @property
    def score_function(self):
        return self.event_score_type.score_function


# TODO: rename submission to workflow and submitted file to workflow_element
# TODO: SubmissionOnCVFold should actually be a workflow element. Saving
# train_pred means that we can input it to the next workflow element
# TODO: implement check
class SubmissionOnCVFold(Model):
    """SubmissionOnCVFold.

    is an instantiation of Submission, to be trained on a data file and a cv
    fold. We don't actually store the trained model in the db (lack of disk and
    pickling issues), so trained submission is not a database column. On the
    other hand, we will store train, valid, and test predictions. In a sense
    substituting CPU time for storage.
    """

    __tablename__ = 'submission_on_cv_folds'

    id = Column(Integer, primary_key=True)

    submission_id = Column(
        Integer, ForeignKey('submissions.id'), nullable=False)
    submission = relationship(
        'Submission', backref=backref(
            'on_cv_folds', cascade="all, delete-orphan"))

    cv_fold_id = Column(
        Integer, ForeignKey('cv_folds.id'), nullable=False)
    cv_fold = relationship(
        'CVFold', backref=backref(
            'submissions', cascade="all, delete-orphan"))

    # filled by cv_fold.get_combined_predictions
    contributivity = Column(Float, default=0.0)
    best = Column(Boolean, default=False)

    # prediction on the full training set, including train and valid points
    # properties train_predictions and valid_predictions will make the slicing
    full_train_y_pred = Column(NumpyType, default=None)
    test_y_pred = Column(NumpyType, default=None)
    train_time = Column(Float, default=0.0)
    valid_time = Column(Float, default=0.0)
    test_time = Column(Float, default=0.0)
    state = Column(submission_states, default='new')
    error_msg = Column(String, default='')

    UniqueConstraint(submission_id, cv_fold_id, name='sc_constraint')

    def __init__(self, submission, cv_fold):
        self.submission = submission
        self.cv_fold = cv_fold
        self.session = inspect(submission).session
        for score in submission.scores:
            submission_score_on_cv_fold = SubmissionScoreOnCVFold(
                submission_on_cv_fold=self, submission_score=score)
            self.session.add(submission_score_on_cv_fold)
        self.reset()

    def __repr__(self):
        repr = 'state = {}, c = {}'\
            ', best = {}'.format(
                self.state, self.contributivity, self.best)
        return repr

    @hybrid_property
    def is_public_leaderboard(self):
        return self.state == 'scored'

    @hybrid_property
    def is_trained(self):
        return self.state in\
            ['trained', 'validated', 'tested', 'validating_error',
             'testing_error', 'scored']

    @hybrid_property
    def is_validated(self):
        return self.state in ['validated', 'tested', 'testing_error', 'scored']

    @hybrid_property
    def is_tested(self):
        return self.state in ['tested', 'scored']

    @hybrid_property
    def is_error(self):
        return (self.state == 'training_error') |\
            (self.state == 'checking_error') |\
            (self.state == 'validating_error') |\
            (self.state == 'testing_error')

    # The following four functions are converting the stored numpy arrays
    # <>_y_pred into Prediction instances
    @property
    def full_train_predictions(self):
        return self.submission.Predictions(y_pred=self.full_train_y_pred)

    @property
    def train_predictions(self):
        return self.submission.Predictions(
            y_pred=self.full_train_y_pred[self.cv_fold.train_is])

    @property
    def valid_predictions(self):
        return self.submission.Predictions(
            y_pred=self.full_train_y_pred[self.cv_fold.test_is])

    @property
    def test_predictions(self):
        return self.submission.Predictions(y_pred=self.test_y_pred)

    @property
    def official_score(self):
        for score in self.scores:
            if self.submission.official_score_name == score.name:
                return score

    def reset(self):
        self.contributivity = 0.0
        self.best = False
        self.full_train_y_pred = None
        self.test_y_pred = None
        self.train_time = 0.0
        self.valid_time = 0.0
        self.test_time = 0.0
        self.state = 'new'
        self.error_msg = ''
        for score in self.scores:
            score.train_score = score.event_score_type.worst
            score.valid_score = score.event_score_type.worst
            score.test_score = score.event_score_type.worst

    def set_error(self, error, error_msg):
        self.reset()
        self.state = error
        self.error_msg = error_msg

    def compute_train_scores(self):
        if self.is_trained:
            true_full_train_predictions =\
                self.submission.event.problem.ground_truths_train()
            for score in self.scores:
                score.train_score = float(score.score_function(
                    true_full_train_predictions, self.full_train_predictions,
                    self.cv_fold.train_is))
        else:
            for score in self.scores:
                score.train_score = score.event_score_type.worst

    def compute_valid_scores(self):
        if self.is_validated:
            true_full_train_predictions =\
                self.submission.event.problem.ground_truths_train()
            for score in self.scores:
                score.valid_score = float(score.score_function(
                    true_full_train_predictions, self.full_train_predictions,
                    self.cv_fold.test_is))
        else:
            for score in self.scores:
                score.valid_score = score.event_score_type.worst

    def compute_test_scores(self):
        if self.is_tested:
            true_test_predictions =\
                self.submission.event.problem.ground_truths_test()
            for score in self.scores:
                score.test_score = float(score.score_function(
                    true_test_predictions, self.test_predictions))
        else:
            for score in self.scores:
                score.test_score = score.event_score_type.worst

    def update(self, detached_submission_on_cv_fold):
        """From trained DetachedSubmissionOnCVFold."""
        self.state = detached_submission_on_cv_fold.state
        if self.is_error:
            self.error_msg = detached_submission_on_cv_fold.error_msg
        else:
            if self.is_trained:
                self.train_time = detached_submission_on_cv_fold.train_time
            if self.is_validated:
                self.valid_time = detached_submission_on_cv_fold.valid_time
                self.full_train_y_pred =\
                    detached_submission_on_cv_fold.full_train_y_pred
            if self.is_tested:
                self.test_time = detached_submission_on_cv_fold.test_time
                self.test_y_pred = detached_submission_on_cv_fold.test_y_pred


class DetachedSubmissionOnCVFold(object):
    """Copy of SubmissionOnCVFold, all the fields we need in train and test.

    It's because SQLAlchemy objects don't persist through
    multiprocessing jobs. Maybe eliminated if we do the parallelization
    differently, though I doubt it.
    """

    def __init__(self, submission_on_cv_fold):
        self.train_is = submission_on_cv_fold.cv_fold.train_is
        self.test_is = submission_on_cv_fold.cv_fold.test_is
        self.full_train_y_pred = submission_on_cv_fold.full_train_y_pred
        self.test_y_pred = submission_on_cv_fold.test_y_pred
        self.state = submission_on_cv_fold.state
        self.name = submission_on_cv_fold.submission.event.name + '/'\
            + submission_on_cv_fold.submission.team.name + '/'\
            + submission_on_cv_fold.submission.name
        self.path = submission_on_cv_fold.submission.path
        self.error_msg = submission_on_cv_fold.error_msg
        self.train_time = submission_on_cv_fold.train_time
        self.valid_time = submission_on_cv_fold.valid_time
        self.test_time = submission_on_cv_fold.test_time
        self.trained_submission = None
        self.workflow =\
            submission_on_cv_fold.submission.event.problem.workflow_object

    def __repr__(self):
        text = 'Submission({}) on fold {}'.format(
            self.name, str(self.train_is)[:10])
        return text


submission_similarity_type = Enum(
    'target_credit',  # credit given by one of the authors of target
    'source_credit',  # credit given by one of the authors of source
    'thirdparty_credit',  # credit given by an independent user
    name='submission_similarity_type'
)


class SubmissionSimilarity(Model):
    __tablename__ = 'submission_similaritys'

    id = Column(Integer, primary_key=True)
    type = Column(submission_similarity_type, nullable=False)
    note = Column(String, default=None)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow())
    similarity = Column(Float, default=0.0)

    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship(
        'User', backref=backref('submission_similaritys'))

    source_submission_id = Column(
        Integer, ForeignKey('submissions.id'))
    source_submission = relationship(
        'Submission', primaryjoin=(
            'SubmissionSimilarity.source_submission_id == Submission.id'),
        backref=backref('sources', cascade='all, delete-orphan'))

    target_submission_id = Column(
        Integer, ForeignKey('submissions.id'))
    target_submission = relationship(
        'Submission', primaryjoin=(
            'SubmissionSimilarity.target_submission_id == Submission.id'),
        backref=backref('targets', cascade='all, delete-orphan'))

    def __repr__(self):
        text = 'type={}, user={}, source={}, target={} '.format(
            self.type, self.user, self.source_submission,
            self.target_submission)
        text += 'similarity={}, timestamp={}'.format(
            self.similarity, self.timestamp)
        return text
