import os
import hashlib
import datetime

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


def _encode_string(text):
    return bytes(text, 'utf-8') if isinstance(text, str) else text


class Submission(Model):
    """Submission table.

    Parameters
    ----------
    name : str
        The submission name.
    event_team : :class:`ramp_database.model.EventTeam`
        The event/team instance.
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.

    Attributes
    ----------
    id : int
        The ID of the table row.
    event_team_id : int
        The event/team ID.
    event_team : :class:`ramp_database.model.EventTeam`
        The event/team instance.
    name : str
        The name of the submission.
    hash_ : string
        A hash to identify the submission.
    files : list of :class:`ramp_database.model.SubmissionFile`
        The list of the files associated with the submission.
    submission_timestamp : datetime
        The date and time when the submission was added to the database.
    sent_to_training_timestamp : datetime
        The date and time when the submission was sent for training.
    training_timestamp : datetime
        The date and time when the training finished.
    contributivity : float
        The contributivity of the submission.
    historical_contributivity : float
        The historical contributivity.
    type : {'live' or 'test'}
        The type of submission.
    state : str
        The state of the submission. For possible states, see the
        ``submission_states`` enum in this module (top of this file).
    error_msg : str
        The error message of the submission.
    is_valid : bool
        Is it a valid submission.
    is_to_ensemble : bool
        Whether to use the submission for the contributivity score.
    is_in_competition : bool
        Whether the submission is used to participate to the comptetition.
    notes : str
        Store any note regarding the submission.
    train_time_cv_mean : float
        The mean of the computation time for a fold on the train data.
    valid_time_cv_mean : float
        The mean of the computation time for a fold on the valid data.
    test_time_cv_mean : float
        The mean of the computation time for a fold on the test data.
    train_time_cv_std : float
        The standard deviation of the computation time for a fold on the train
        data.
    valid_time_cv_std : float
        The standard deviation of the computation time for a fold on the valid
        data.
    test_time_cv_std : float
        The standard deviation of the computation time for a fold on the test
        data.
    max_ram : float
        The maximum amount of RAM consumed during training.
    historical_contributivitys : list of \
:class:`ramp_database.model.HistoricalContributivity`
        A back-reference of the historical contributivities for the submission.
    scores : list of :class:`ramp_database.model.SubmissionScore`
        A back-reference of scores for the submission.
    files : list of :class:`ramp_database.model.SubmissionFile`
        A back-reference of files attached to the submission.
    on_cv_folds : list of :class:`ramp_database.model.SubmissionOnCVFold`
        A back-reference of the CV fold for this submission.
    """
    __tablename__ = 'submissions'

    id = Column(Integer, primary_key=True)

    event_team_id = Column(Integer, ForeignKey('event_teams.id'),
                           nullable=False)
    event_team = relationship('EventTeam',
                              backref=backref('submissions',
                                              cascade='all, delete-orphan'))

    name = Column(String(20), nullable=False)
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

    def __init__(self, name, event_team, session=None):
        self.name = name
        self.event_team = event_team
        self.session = inspect(event_team).session
        sha_hasher = hashlib.sha1()
        sha_hasher.update(_encode_string(self.event.name))
        sha_hasher.update(_encode_string(self.team.name))
        sha_hasher.update(_encode_string(self.name))
        # We considered using the id, but then it will be given away in the
        # url which is maybe not a good idea.
        self.hash_ = '{}'.format(sha_hasher.hexdigest())
        self.submission_timestamp = datetime.datetime.utcnow()
        if session is None:
            event_score_types = \
                (EventScoreType.query.filter_by(event=event_team.event))
        else:
            event_score_types = (session.query(EventScoreType)
                                        .filter(EventScoreType.event ==
                                                event_team.event)
                                        .all())
        for event_score_type in event_score_types:
            submission_score = SubmissionScore(
                submission=self, event_score_type=event_score_type)
            self.session.add(submission_score)
        self.reset()

    def __str__(self):
        return 'Submission({}/{}/{})'.format(
            self.event.name, self.team.name, self.name)

    def __repr__(self):
        return ('Submission(event_name={}, team_name={}, name={}, files={}, '
                'state={}, train_time={})'
                .format(self.event.name, self.team.name, self.name,
                        self.files, self.state, self.train_time_cv_mean))

    @hybrid_property
    def team(self):
        """str: The team name."""
        return self.event_team.team

    @hybrid_property
    def event(self):
        """:class:`ramp_database.model.Event`: The event associated with the
        submission."""
        return self.event_team.event

    @property
    # This will work only with Flask
    def official_score_function(self):
        """callable: The scoring function."""
        return self.event.official_score_function

    @property
    def official_score_name(self):
        """str: The name of the default score."""
        return self.event.official_score_name

    @property
    def official_score(self):
        """:class:`ramp_database.model.SubmissionScore`: The official score."""
        score_dict = {score.score_name: score for score in self.scores}
        return score_dict[self.official_score_name]

    @property
    def score_types(self):
        """list of :class:`ramp_database.model.EventScoreType`: All the scores used
        for the submissions."""
        return self.event.score_types

    @property
    def Predictions(self):
        """:class:`rampwf.prediction_types`: The predictions used for the
        problem."""
        return self.event.Predictions

    @hybrid_property
    def is_not_sandbox(self):
        """bool: Whether the submission is not a sandbox."""
        return self.name != self.event.ramp_sandbox_name

    @hybrid_property
    def is_error(self):
        """bool: Whether the training of the submission failed."""
        return 'error' in self.state

    @hybrid_property
    def is_new(self):
        """bool: Whether the submission is a new submission."""
        return (self.state in ['new', 'training', 'sent_to_training'] and
                self.is_not_sandbox)

    @hybrid_property
    def is_public_leaderboard(self):
        """bool: Whether the submission is part of the public leaderboard."""
        return (self.is_not_sandbox and self.is_valid and
                (self.state == 'scored'))

    @hybrid_property
    def is_private_leaderboard(self):
        """bool: Whether the submission is part of the private leaderboard."""
        return (self.is_not_sandbox and self.is_valid and
                (self.state == 'scored'))

    @property
    def path(self):
        """str: The path to the submission."""
        return os.path.join(self.event.path_ramp_submissions, self.basename)

    @property
    def basename(self):
        """str: The base name of the submission."""
        return 'submission_' + '{:09d}'.format(self.id)

    @property
    def module(self):
        """str: Path of the submission as a module."""
        return self.path.lstrip('./').replace('/', '.')

    @property
    def f_names(self):
        """list of str: File names of a submission."""
        return [file.f_name for file in self.files]

    @property
    def link(self):
        """str: Unique link to the first submission file."""
        return self.files[0].link

    @property
    def full_name_with_link(self):
        """str: HTML hyperlink to the first submission file with event, team,
        and submission information.

        The hyperlink forward to the first submission file while the text
        corresponds to the event, team, and submission name.
        """
        return '<a href={}>{}/{}/{}</a>'.format(
            self.link, self.event.name, self.team.name, self.name[:20])

    @property
    def name_with_link(self):
        """str: HTML hyperlink to the first submission file with submission
        information.

        The hyperlink forward to the first submission file while the text
        corresponds to submission name.
        """
        return '<a href={}>{}</a>'.format(self.link, self.name[:20])

    @property
    def state_with_link(self):
        """str: HTML hyperlink to the state file to report error."""
        return '<a href=/{}>{}</a>'.format(
            os.path.join(self.hash_, 'error.txt'), self.state)

    def ordered_scores(self, score_names):
        """Generator yielding :class:`ramp_database.model.SubmissionScore`.

        Ordered according to ``score_names``. Called by
        :func:`ramp_database.tools.leaderboard.get_public_leaderboard` and
        :func:`ramp_database.tools.get_private_leaderboard`, making sure scores
        are listed in the correct column.

        Parameters
        ----------
        score_names : list of str
            Name of the scores.

        Returns
        -------
        scores : generator of \
:class:`ramp_database.model.submission.SubmissionScore``
            Generate a scoring instance.
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

    def set_state(self, state, session=None):
        """Set the state of the submission and of each CV fold.

        Parameters
        ----------
        state : str
            The state of the new submission.
        """
        self.state = state

        if state == "sent_to_training":
            self.sent_to_training_timestamp = datetime.datetime.utcnow()
        elif state == "training":
            self.training_timestamp = datetime.datetime.utcnow()

        if session is None:
            all_cv_folds = self.on_cv_folds
        else:
            all_cv_folds = (session.query(SubmissionOnCVFold)
                                   .filter_by(submission_id=self.id)
                                   .all())
            all_cv_folds = sorted(all_cv_folds, key=lambda x: x.id)
        for submission_on_cv_fold in all_cv_folds:
            submission_on_cv_fold.state = state

    def reset(self):
        """Reset the submission to an initial stage.

        The contributivity, state, error, and scores will be reset to initial
        values.
        """
        self.contributivity = 0.0
        self.state = 'new'
        self.error_msg = ''
        for score in self.scores:
            score.valid_score_cv_bag = score.event_score_type.worst
            score.test_score_cv_bag = score.event_score_type.worst
            score.valid_score_cv_bags = None
            score.test_score_cv_bags = None

    def set_error(self, error, error_msg, session=None):
        """Fail the submission as well as the CV folds.

        Parameters
        ----------
        error : str
            The error state of the submission and each fold.
        error_msg : str
            The associated error message for the submission and each fold.

        Notes
        -----
        Setting the error will first reset the submission.
        """
        self.reset()
        self.state = error
        self.error_msg = error_msg
        if session is None:
            all_cv_folds = self.on_cv_folds
        else:
            all_cv_folds = (session.query(SubmissionOnCVFold)
                                   .filter_by(submission_id=self.id)
                                   .all())
            all_cv_folds = sorted(all_cv_folds, key=lambda x: x.id)
        for submission_on_cv_fold in all_cv_folds:
            submission_on_cv_fold.set_error(error, error_msg)

    # contributivity could be a property but then we could not query on it
    def set_contributivity(self, session=None):
        """Compute the contributivity of a submission.

        Notes
        -----
        The contributivity is computed only id the submission is public and
        valid and this is not the sandbox submission.
        """
        self.contributivity = 0.0
        if self.is_public_leaderboard:
            # we share a unit of 1. among folds
            if session is None:
                all_cv_folds = self.on_cv_folds
            else:
                all_cv_folds = (session.query(SubmissionOnCVFold)
                                       .filter_by(submission_id=self.id)
                                       .all())
                all_cv_folds = sorted(all_cv_folds, key=lambda x: x.id)
            unit_contributivity = 1. / len(all_cv_folds)
            for submission_on_cv_fold in all_cv_folds:
                self.contributivity += (unit_contributivity *
                                        submission_on_cv_fold.contributivity)

    def set_state_after_training(self, session=None):
        """Set the state of a submission depending of the state of the fold
        after training.
        """
        self.training_timestamp = datetime.datetime.utcnow()
        if session is None:
            all_cv_folds = self.on_cv_folds
        else:
            all_cv_folds = (session.query(SubmissionOnCVFold)
                                   .filter_by(submission_id=self.id)
                                   .all())
            all_cv_folds = sorted(all_cv_folds, key=lambda x: x.id)
        states = [submission_on_cv_fold.state
                  for submission_on_cv_fold in all_cv_folds]
        if all(state == 'tested' for state in states):
            self.state = 'tested'
        elif all(state in ['tested', 'validated'] for state in states):
            self.state = 'validated'
        elif all(state in ['tested', 'validated', 'trained']
                 for state in states):
            self.state = 'trained'
        elif any(state == 'training_error' for state in states):
            self.state = 'training_error'
            i = states.index('training_error')
            self.error_msg = all_cv_folds[i].error_msg
        elif any(state == 'validating_error' for state in states):
            self.state = 'validating_error'
            i = states.index('validating_error')
            self.error_msg = all_cv_folds[i].error_msg
        elif any(state == 'testing_error' for state in states):
            self.state = 'testing_error'
            i = states.index('testing_error')
            self.error_msg = all_cv_folds[i].error_msg
        if 'error' not in self.state:
            self.error_msg = ''


class SubmissionScore(Model):
    """SubmissionScore table.

    Attributes
    ----------
    id : int
        The ID of the row table.
    submission_id : int
        The ID of the associated submission.
    submission : :class:`ramp_database.model.Submission`
        The submission instance associated.
    event_score_type_id : int
        The ID of the event/score type associated.
    event_score_type : :class:`ramp_database.model.EventScoreType`
        The event/score type instance associated.
    valid_score_cv_bag : float
        The validation bagged scores.
    test_score_cv_bag : float
        The testing bagged scores.
    valid_score_cv_bags : ndarray
        The partial validation scores for all CV bags.
    test_score_cv_bags : ndarray
        The partial testing scores for all CV bags.
    on_cv_folds : list of :class:`ramp_database.model.SubmissionScoreOnCVFold`
        A back-reference the CV fold associated with the score.
    """
    __tablename__ = 'submission_scores'

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey('submissions.id'),
                           nullable=False)
    submission = relationship('Submission',
                              backref=backref('scores',
                                              cascade='all, delete-orphan'))

    event_score_type_id = Column(Integer, ForeignKey('event_score_types.id'),
                                 nullable=False)
    event_score_type = relationship('EventScoreType',
                                    backref=backref('submissions'))

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
        """str: The name of the score."""
        return self.event_score_type.name

    @property
    def score_function(self):
        """callable: The function used to score."""
        return self.event_score_type.score_function

    # default display precision in n_digits
    @property
    def precision(self):
        """int: The numerical precision of the associated score."""
        return self.event_score_type.precision


# TODO: we should have a SubmissionWorkflowElementType table, describing the
# type of files we are expecting for a given RAMP. Fast unit test should be
# set up there, and each file should be unit tested right after submission.
# Cosmetic: Perhaps mark which file the leaderboard link should point to (right
# now it is set to the first file in the list which is arbitrary).
# We will also have to handle auxiliary files (like CSVs or other classes).
# User interface could have a single submission form with a menu containing
# the file names for a given ramp + an "other" field when users will have to
# name their files
class SubmissionFile(Model):
    """SubmissionFile table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    submission_id : int
        The ID of the associated submission.
    submission : :class:`ramp_database.model.Submission`
        The submission instance associated.
    workflow_element_id : int
        The ID of the associated workflow element.
    workflow_element : :class:`ramp_database.model.WorkflowElement`
        The workflow element associated with the submission.
    submission_file_type_extension_id : int
        The ID of the associated submission file type extension.
    submission_file_type_extension : \
:class:`ramp_database.model.SubmissionFileTypeExtension`
        The associated submission file type extension instance.
    """
    __tablename__ = 'submission_files'

    id = Column(Integer, primary_key=True)
    submission_id = Column(Integer, ForeignKey('submissions.id'),
                           nullable=False)
    submission = relationship('Submission',
                              backref=backref('files',
                                              cascade='all, delete-orphan'))

    workflow_element_id = Column(Integer, ForeignKey('workflow_elements.id'),
                                 nullable=False)
    workflow_element = relationship('WorkflowElement',
                                    backref=backref('submission_files'))

    submission_file_type_extension_id = Column(
        Integer, ForeignKey('submission_file_type_extensions.id'),
        nullable=False
    )
    submission_file_type_extension = relationship(
        'SubmissionFileTypeExtension', backref=backref('submission_files')
    )

    def __repr__(self):
        return ('SubmissionFile(name={}, type={}, extension={}, path={})'
                .format(self.name, self.type, self.extension, self.path))

    @property
    def is_editable(self):
        """bool: Whether the submission file is from an editable format on the
        frontend."""
        return self.workflow_element.is_editable

    @property
    def extension(self):
        """str: The extension of the file format."""
        return self.submission_file_type_extension.extension.name

    @property
    def type(self):
        """str: The workflow type associated with the file."""
        return self.workflow_element.type

    @property
    def name(self):
        """str: The name of the workflow element."""
        return self.workflow_element.name

    @property
    def f_name(self):
        """str: The corresponding file name."""
        return self.type + '.' + self.extension

    @property
    def link(self):
        """str: A unique link to the file. The hash is generated by the
        Submission instance."""
        return '/' + os.path.join(self.submission.hash_, self.f_name)

    @property
    def path(self):
        """str: The path to the file in the deployment directory."""
        return os.path.join(self.submission.path, self.f_name)

    @property
    def name_with_link(self):
        """str: The HTML hyperlink of the name of the submission file."""
        return '<a href="' + self.link + '">' + self.name + '</a>'

    def get_code(self):
        """str: Get the content of the file."""
        with open(self.path) as f:
            code = f.read()
        return code

    def set_code(self, code):
        """Set the content of the submission file.

        Parameters
        ----------
        code : str
            The code to write into the submission file.
        """
        with open(self.path, 'w') as f:
            f.write(code)


class SubmissionFileTypeExtension(Model):
    """SubmissionFileTypeExtension table.

    This a many-to-many relationship between the SubmissionFileType and
    Extension.

    Attributes
    ----------
    id : int
        The ID of the table row.
    type_id : int
        The ID of the submission file type.
    type : :class:`ramp_database.model.SubmissionFileType`
        The submission file type instance.
    extension_id : int
        The ID of the extension.
    extension : :class:`ramp_database.model.Extension`
        The file extension instance.
    submission_files : list of \
:class:`ramp_database.model.SubmissionFileTypeExtension`
        A back-reference to the submission files related to the type extension.
    """
    __tablename__ = 'submission_file_type_extensions'

    id = Column(Integer, primary_key=True)

    type_id = Column(Integer, ForeignKey('submission_file_types.id'),
                     nullable=False)
    type = relationship('SubmissionFileType', backref=backref('extensions'))

    extension_id = Column(Integer, ForeignKey('extensions.id'), nullable=False)
    extension = relationship('Extension',
                             backref=backref('submission_file_types'))

    UniqueConstraint(type_id, extension_id, name='we_constraint')

    @property
    def file_type(self):
        """str: The name of the file type."""
        return self.type.name

    @property
    def extension_name(self):
        """str: The name of the file extension."""
        return self.extension.name


class SubmissionFileType(Model):
    """SubmissionFileType table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the submission file type.
    is_editable : bool
        Whether or not this type of file is editable on the frontend.
    max_size : int
        The maximum size of this file type.
    workflow_element_types : list of \
:class:`ramp_database.model.WorkflowElementType`
        A back-reference to the workflow element type for this submission file
        type.
    """
    __tablename__ = 'submission_file_types'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    is_editable = Column(Boolean, default=True)
    max_size = Column(Integer, default=None)


class Extension(Model):
    """Extension table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the extension.
    submission_file_types : list of \
:class:`ramp_database.model.SubmissionFileTypeExtension`
        A back-reference to the submission file types for this extension.
    """
    __tablename__ = 'extensions'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)


class SubmissionScoreOnCVFold(Model):
    """SubmissionScoreOnCVFold table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    submission_on_cv_fold_id : int
        The ID of the CV fold.
    submission_on_cv_fold : :class:`ramp_database.model.SubmissionOnCVFold`
        The submission on CV fold instance.
    submission_score_id : int
        The ID of the submission score.
    submission_score : :class:`ramp_database.model.SubmissionScore`
        The submission score instance.
    train_score : float
        The training score on the fold.
    valid_score : float
        The validation score on the fold.
    test_score : float
        The testing score on the fold.
    """
    __tablename__ = 'submission_score_on_cv_folds'

    id = Column(Integer, primary_key=True)
    submission_on_cv_fold_id = Column(
        Integer, ForeignKey('submission_on_cv_folds.id'), nullable=False
    )
    submission_on_cv_fold = relationship(
        'SubmissionOnCVFold',
        backref=backref('scores', cascade='all, delete-orphan')
    )

    submission_score_id = Column(Integer, ForeignKey('submission_scores.id'),
                                 nullable=False)
    submission_score = relationship(
        'SubmissionScore',
        backref=backref('on_cv_folds', cascade='all, delete-orphan')
    )

    train_score = Column(Float)
    valid_score = Column(Float)
    test_score = Column(Float)

    UniqueConstraint(
        submission_on_cv_fold_id, submission_score_id, name='ss_constraint')

    @property
    def name(self):
        """str: The name of the score."""
        return self.event_score_type.name

    @property
    def event_score_type(self):
        """:class:`EventScoreType`: The event/score type instance."""
        return self.submission_score.event_score_type

    @property
    def score_function(self):
        """callable: the scoring function."""
        return self.event_score_type.score_function


# TODO: rename submission to workflow and submitted file to workflow_element
# TODO: SubmissionOnCVFold should actually be a workflow element. Saving
# train_pred means that we can input it to the next workflow element
# TODO: implement check
class SubmissionOnCVFold(Model):
    """SubmissionOnCVFold.

    Parameters
    ----------
    submission : :class:`ramp_database.model.Submission`
        The submission used.
    cv_fold : :class:`ramp_database.model.CVFold`
        The fold to associate with the submission.

    Attributes
    ----------
    id : int
        The ID of the table row.
    submission_id : int
        The ID of the submission.
    submission : :class:`ramp_database.model.Submission`
        The submission instance.
    cv_fold_id : int
        The ID of the CV fold.
    cv_fold : :class:`ramp_database.model.CVFold`
        The CV fold instance.
    contributivity : float
        The contributivity of the submission.
    best : bool
        Whether or not the submission is the best.
    full_train_y_pred : ndarray
        Predictions on the full training set.
    test_y_pred : ndarray
        Predictions on the testing set.
    train_time : float
        Computation time for the training set.
    valid_time : float
        Computation time for the validation set.
    test_time : float
        Computation time for the testing set.
    state : str
        State of of the submission on this fold.
    error_msg : str
        Error message in case of failing submission.
    scores : list of :class:`ramp_database.model.SubmissionScoreOnCVFold`
        A back-reference on the scores for this fold.

    Notes
    -----
    SubmissionOnCVFold is an instantiation of Submission, to be trained on a
    data file and a cv fold. We don't actually store the trained model in the
    db (lack of disk and pickling issues), so trained submission is not a
    database column. On the other hand, we will store train, valid, and test
    predictions. In a sense substituting CPU time for storage.
    """

    __tablename__ = 'submission_on_cv_folds'

    id = Column(Integer, primary_key=True)

    submission_id = Column(Integer, ForeignKey('submissions.id'),
                           nullable=False)
    submission = relationship('Submission',
                              backref=backref('on_cv_folds',
                                              cascade="all, delete-orphan"))

    cv_fold_id = Column(Integer, ForeignKey('cv_folds.id'), nullable=False)
    cv_fold = relationship('CVFold',
                           backref=backref('submissions',
                                           cascade="all, delete-orphan"))

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
        return ('state = {}, c = {}, best = {}'
                .format(self.state, self.contributivity, self.best))

    @hybrid_property
    def is_public_leaderboard(self):
        """bool: Whether or not the submission is scored and ready to be on
        the public leaderboard."""
        return self.state == 'scored'

    @hybrid_property
    def is_trained(self):
        """bool: Whether or not the submission was trained."""
        return self.state in ('trained', 'validated', 'tested',
                              'validating_error', 'testing_error', 'scored')

    @hybrid_property
    def is_validated(self):
        """bool: Whether or not the submission was validated."""
        return self.state in ('validated', 'tested', 'testing_error', 'scored')

    @hybrid_property
    def is_tested(self):
        """bool: Whether or not the submission was tested."""
        return self.state in ('tested', 'scored')

    @hybrid_property
    def is_error(self):
        """bool: Whether or not the submission failed at one of the stage."""
        return 'error' in self.state

    # The following four functions are converting the stored numpy arrays
    # <>_y_pred into Prediction instances
    @property
    def full_train_predictions(self):
        """:class:`rampwf.prediction_types.Predictions`: Training
        predictions."""
        return self.submission.Predictions(y_pred=self.full_train_y_pred)

    @property
    def train_predictions(self):
        """:class:`rampwf.prediction_types.Predictions`: Training
        predictions."""
        return self.submission.Predictions(
            y_pred=self.full_train_y_pred[self.cv_fold.train_is])

    @property
    def valid_predictions(self):
        """:class:`rampwf.prediction_types.Predictions`: Validation
        predictions."""
        return self.submission.Predictions(
            y_pred=self.full_train_y_pred[self.cv_fold.test_is])

    @property
    def test_predictions(self):
        """:class:`rampwf.prediction_types.Predictions`: Testing
        predictions."""
        return self.submission.Predictions(y_pred=self.test_y_pred)

    @property
    def official_score(self):
        """:class:`ramp_database.model.SubmissionScoreOnCVFold`: The official score
        used for the submission."""
        for score in self.scores:
            if self.submission.official_score_name == score.name:
                return score

    def reset(self):
        """Reset the submission on CV fold to an initial stage.

        The contributivity, state, error, and scores will be reset to initial
        values.
        """
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
        """Fail the CV fold.

        Parameters
        ----------
        error : str
            The error state of the submission and each fold.
        error_msg : str
            The associated error message for the submission and each fold.

        Notes
        -----
        Setting the error will first reset the submission.
        """
        self.reset()
        self.state = error
        self.error_msg = error_msg

    def compute_train_scores(self):
        """Compute all training scores."""
        if self.is_trained:
            true_full_train_predictions = \
                self.submission.event.problem.ground_truths_train()
            for score in self.scores:
                score.train_score = float(score.score_function(
                    true_full_train_predictions,
                    self.full_train_predictions,
                    self.cv_fold.train_is))
        else:
            for score in self.scores:
                score.train_score = score.event_score_type.worst

    def compute_valid_scores(self):
        """Compute all validating scores."""
        if self.is_validated:
            true_full_train_predictions = \
                self.submission.event.problem.ground_truths_train()
            for score in self.scores:
                score.valid_score = float(score.score_function(
                    true_full_train_predictions,
                    self.full_train_predictions,
                    self.cv_fold.test_is))
        else:
            for score in self.scores:
                score.valid_score = score.event_score_type.worst

    def compute_test_scores(self):
        """Compute all testing scores."""
        if self.is_tested:
            true_test_predictions = \
                self.submission.event.problem.ground_truths_test()
            for score in self.scores:
                score.test_score = float(score.score_function(
                    true_test_predictions,
                    self.test_predictions))
        else:
            for score in self.scores:
                score.test_score = score.event_score_type.worst

    def update(self, detached_submission_on_cv_fold):
        """Update the submission on CV Fold from a detached submission.

        Parameters
        ----------
        detached_submission_on_cv_fold : \
:class:`ramp_database.model.DetachedSubmissionOnCVFold`
            The detached submission from which we will update the current
            submission.
        """
        self.state = detached_submission_on_cv_fold.state
        if self.is_error:
            self.error_msg = detached_submission_on_cv_fold.error_msg
        else:
            if self.is_trained:
                self.train_time = detached_submission_on_cv_fold.train_time
            if self.is_validated:
                self.valid_time = detached_submission_on_cv_fold.valid_time
                self.full_train_y_pred = \
                    detached_submission_on_cv_fold.full_train_y_pred
            if self.is_tested:
                self.test_time = detached_submission_on_cv_fold.test_time
                self.test_y_pred = detached_submission_on_cv_fold.test_y_pred


class DetachedSubmissionOnCVFold:
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
        self.name = (submission_on_cv_fold.submission.event.name + '/' +
                     submission_on_cv_fold.submission.team.name + '/' +
                     submission_on_cv_fold.submission.name)
        self.path = submission_on_cv_fold.submission.path
        self.error_msg = submission_on_cv_fold.error_msg
        self.train_time = submission_on_cv_fold.train_time
        self.valid_time = submission_on_cv_fold.valid_time
        self.test_time = submission_on_cv_fold.test_time
        self.trained_submission = None
        self.workflow = \
            submission_on_cv_fold.submission.event.problem.workflow_object

    def __repr__(self):
        return ('Submission({}) on fold {}'
                .format(self.name, str(self.train_is)[:10]))


submission_similarity_type = Enum(
    'target_credit',  # credit given by one of the authors of target
    'source_credit',  # credit given by one of the authors of source
    'thirdparty_credit',  # credit given by an independent user
    name='submission_similarity_type'
)


class SubmissionSimilarity(Model):
    """SubmissionSimilarity table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    type : str
        The type of similarity.
    note : str
        Note about the similarity.
    timestamp : datetime
        The date and time of the submission.
    similarity : float
        The similarity index.
    user_id : int
        The ID of the user.
    user : :class:`ramp_database.model.User`
        The user instance.
    source_submission_id : int
        The ID of the submission used as source.
    source_submission : :class:`ramp_database.model.Submission`
        The source submission instance.
    target_submission_id : int
        The ID of the submission used as target.
    target_submission : :class:`ramp_database.model.Submission`
        The target submission instance.
    """
    __tablename__ = 'submission_similaritys'

    id = Column(Integer, primary_key=True)
    type = Column(submission_similarity_type, nullable=False)
    note = Column(String, default=None)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow())
    similarity = Column(Float, default=0.0)

    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship('User',
                        backref=backref('submission_similaritys',
                                        cascade='all, delete-orphan'))

    source_submission_id = Column(Integer, ForeignKey('submissions.id'))
    source_submission = relationship(
        'Submission', primaryjoin=(
            'SubmissionSimilarity.source_submission_id == Submission.id'),
        backref=backref('sources', cascade='all, delete-orphan')
    )

    target_submission_id = Column(Integer, ForeignKey('submissions.id'))
    target_submission = relationship(
        'Submission', primaryjoin=(
            'SubmissionSimilarity.target_submission_id == Submission.id'),
        backref=backref('targets', cascade='all, delete-orphan'))

    def __repr__(self):
        text = ('type={}, user={}, source={}, target={} '
                .format(self.type, self.user, self.source_submission,
                        self.target_submission))
        text += 'similarity={}, timestamp={}'.format(self.similarity,
                                                     self.timestamp)
        return text
