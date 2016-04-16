import os
import zlib
import hashlib
import logging
import datetime
import numpy as np
from flask import request
from importlib import import_module
from sqlalchemy.ext.hybrid import hybrid_property

from databoard import db
import databoard.config as config

# Training set table
# Problem table
# Ramp table that connects all (training set, problem, cv, specific)
# specific should be cut at least into problem-specific, data and cv-specific,
# and ramp (event) specific files

logger = logging.getLogger('databoard')


class NumpyType(db.TypeDecorator):
    """Storing zipped numpy arrays."""

    impl = db.LargeBinary

    def process_bind_param(self, value, dialect):
        # we convert the initial value into np.array to handle None and lists
        return zlib.compress(np.array(value).dumps())

    def process_result_value(self, value, dialect):
        return np.loads(zlib.decompress(value))


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)
    hashed_password = db.Column(db.String, nullable=False)
    lastname = db.Column(db.String(256), nullable=False)
    firstname = db.Column(db.String(256), nullable=False)
    email = db.Column(db.String(256), nullable=False, unique=True)
    linkedin_url = db.Column(db.String(256), default=None)
    twitter_url = db.Column(db.String(256), default=None)
    facebook_url = db.Column(db.String(256), default=None)
    google_url = db.Column(db.String(256), default=None)
    github_url = db.Column(db.String(256), default=None)
    website_url = db.Column(db.String(256), default=None)
    hidden_notes = db.Column(db.String, default=None)
    bio = db.Column(db.String(1024), default=None)
    is_want_news = db.Column(db.Boolean, default=True)
    access_level = db.Column(db.Enum(
        'admin', 'user', 'asked'), default='asked')  # 'asked' needs approval
    signup_timestamp = db.Column(db.DateTime, nullable=False)

    # Flask-Login fields
    is_authenticated = db.Column(db.Boolean, default=True)
    is_active = db.Column(db.Boolean, default=True)

    def __init__(self, name, hashed_password, lastname, firstname, email,
                 access_level='user', hidden_notes='', linkedin_url='',
                 twitter_url='', facebook_url='', google_url='', github_url='',
                 website_url='', bio='', is_want_news=True):
        self.name = name
        self.hashed_password = hashed_password
        self.lastname = lastname
        self.firstname = firstname
        self.email = email
        self.access_level = access_level
        self.hidden_notes = hidden_notes
        self.signup_timestamp = datetime.datetime.utcnow()
        self.linkedin_url = linkedin_url
        self.twitter_url = twitter_url
        self.facebook_url = facebook_url
        self.google_url = google_url
        self.github_url = github_url
        self.website_url = website_url
        self.bio = bio
        self.is_want_news = is_want_news

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        try:
            return unicode(self.id)  # python 2
        except NameError:
            return str(self.id)  # python 3

    def __str__(self):
        str_ = 'User({})'.format(self.name)
#        str_ = 'User({}, admined=['.format(self.name)
#        str_ += string.join([team.name for team in self.admined_teams], ', ')
#        str_ += '])'
        return str_

    def __repr__(self):
        repr = '''User(name={}, lastname={}, firstname={}, email={},
                  admined_teams={})'''.format(
            self.name, self.lastname, self.firstname, self.email,
            self.admined_teams)
        return repr


class Team(db.Model):
    __tablename__ = 'teams'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)

    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    admin = db.relationship('User', backref=db.backref('admined_teams'))

    # initiator asks for merge, acceptor accepts
    initiator_id = db.Column(
        db.Integer, db.ForeignKey('teams.id'), default=None)
    initiator = db.relationship(
        'Team', primaryjoin=('Team.initiator_id == Team.id'), uselist=False)

    acceptor_id = db.Column(
        db.Integer, db.ForeignKey('teams.id'), default=None)
    acceptor = db.relationship(
        'Team', primaryjoin=('Team.acceptor_id == Team.id'), uselist=False)

    creation_timestamp = db.Column(db.DateTime, nullable=False)

    def __init__(self, name, admin, initiator=None, acceptor=None):
        self.name = name
        self.admin = admin
        self.initiator = initiator
        self.acceptor = acceptor
        self.creation_timestamp = datetime.datetime.utcnow()

    def __str__(self):
        str_ = 'Team({})'.format(self.name)
        return str_

    def __repr__(self):
        repr = '''Team(name={}, admin_name={},
                  initiator={}, acceptor={})'''.format(
            self.name, self.admin.name, self.initiator, self.acceptor)
        return repr


def get_team_members(team):
    if team.initiator is not None:
        # "yield from" in Python 3.3
        for member in get_team_members(team.initiator):
            yield member
        for member in get_team_members(team.acceptor):
            yield member
    else:
        yield team.admin


def get_n_team_members(team):
    return len(list(get_team_members(team)))


def get_user_teams(user):
    teams = Team.query.all()
    for team in teams:
        if user in get_team_members(team):
            yield team


def get_user_event_teams(event_name, user_name):
    event = Event.query.filter_by(name=event_name).one()
    user = User.query.filter_by(name=user_name).one()
    event_teams = EventTeam.query.filter_by(event=event).all()
    for event_team in event_teams:
        if user in get_team_members(event_team.team):
            yield event_team


def get_n_user_teams(user):
    return len(get_user_teams(user))


# a given RAMP problem, like iris or variable_stars
class Problem(db.Model):
    __tablename__ = 'problems'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False, unique=True)

    workflow_id = db.Column(
        db.Integer, db.ForeignKey('workflows.id'), nullable=False)
    workflow = db.relationship(
        'Workflow', backref=db.backref('problems'))

    def __init__(self, name):
        self.name = name
        self.workflow = Workflow.query.filter_by(
            name=self.module.workflow_name).one()

    def __repr__(self):
        repr = 'Problem({})'.format(self.name)
        return repr

    @property
    def module(self):
        return import_module('.' + self.name, config.problems_module)

    @property
    def prediction(self):
        return self.module.prediction

    def true_predictions_train(self):
        _, y_train = self.module.get_train_data()
        return self.prediction.Predictions(y_true=y_train)

    def true_predictions_test(self):
        _, y_test = self.module.get_test_data()
        return self.prediction.Predictions(y_true=y_test)

    def true_predictions_valid(self, test_is):
        _, y_train = self.module.get_train_data()
        return self.prediction.Predictions(y_true=y_train[test_is])

    @property
    def train_submission(self):
        return self.workflow.train_submission

    @property
    def test_submission(self):
        return self.workflow.test_submission


class ScoreType(db.Model):
    __tablename__ = 'score_types'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False, unique=True)
    is_lower_the_better = db.Column(db.Boolean)
    minimum = db.Column(db.Float)
    maximum = db.Column(db.Float)

    def __repr__(self):
        repr = 'ScoreType(name={})'.format(self.name)
        return repr

    @property
    def module(self):
        return import_module('.' + self.name, config.score_types_module)

    @property
    def score_function(self):
        return self.module.score_function

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    # default display precision in n_digits
    @property
    def precision(self):
        return self.module.precision


# a given RAMP event, like iris_test or M2_data_science_2015_variable_stars
class Event(db.Model):
    __tablename__ = 'events'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False, unique=True)

    problem_id = db.Column(
        db.Integer, db.ForeignKey('problems.id'), nullable=False)
    problem = db.relationship('Problem', backref=db.backref('events'))

    max_members_per_team = db.Column(db.Integer, default=1)
    # max number of submissions in Caruana's ensemble
    max_n_ensemble = db.Column(db.Integer, default=80)
    is_send_trained_mails = db.Column(db.Boolean, default=True)
    is_send_submitted_mails = db.Column(db.Boolean, default=True)
    is_public = db.Column(db.Boolean, default=False)
    is_controled_signup = db.Column(db.Boolean, default=True)

    min_duration_between_submissions = db.Column(db.Integer, default=15 * 60)
    opening_timestamp = db.Column(
        db.DateTime, default=datetime.datetime(2000, 1, 1, 0, 0, 0))
    # before links to submissions in leaderboard are not alive
    public_opening_timestamp = db.Column(
        db.DateTime, default=datetime.datetime(2000, 1, 1, 0, 0, 0))
    closing_timestamp = db.Column(
        db.DateTime, default=datetime.datetime(4000, 1, 1, 0, 0, 0))

    # the index of the score in self.event_score_types which is used for
    # ensembling and contributivity. The default is 0 (first in the list).
    official_score_index = db.Column(db.Integer, default=0)

    combined_combined_valid_score = db.Column(db.Float, default=None)
    combined_combined_test_score = db.Column(db.Float, default=None)
    combined_foldwise_valid_score = db.Column(db.Float, default=None)
    combined_foldwise_test_score = db.Column(db.Float, default=None)

    def __init__(self, name):
        self.name = name
        self.problem = Problem.query.filter_by(
            name=self.module.problem_name).one()

    def __repr__(self):
        repr = 'Event({})'.format(self.name)
        return repr

    @property
    def module(self):
        return import_module('.' + self.name, config.events_module)

    @property
    def title(self):
        return self.module.event_title

    @property
    def prediction(self):
        return self.problem.prediction

    @property
    def official_score_function(self):
        return self.score_types[
            self.official_score_index].score_type.score_function

    @property
    def official_score_type(self):
        return self.score_types[self.official_score_index].score_type

    @property
    def n_cv(self):
        return self.module.n_cv

    @property
    def train_submission(self):
        return self.problem.train_submission

    @property
    def test_submission(self):
        return self.problem.test_submission

    @property
    def combined_combined_valid_score_str(self):
        return None if self.combined_foldwise_valid_score is None else str(
            round(self.combined_combined_valid_score, self.score_precision))

    @property
    def combined_combined_test_score_str(self):
        return None if self.combined_combined_test_score is None else str(
            round(self.combined_combined_test_score, self.score_precision))

    @property
    def combined_foldwise_valid_score_str(self):
        return None if self.combined_foldwise_valid_score is None else str(
            round(self.combined_foldwise_valid_score, self.score_precision))

    @property
    def combined_foldwise_test_score_str(self):
        return None if self.combined_foldwise_test_score is None else str(
            round(self.combined_foldwise_test_score, self.score_precision))

    @property
    def is_open(self):
        now = datetime.datetime.utcnow()
        return now > self.opening_timestamp and now < self.closing_timestamp

    @property
    def is_public_open(self):
        now = datetime.datetime.utcnow()
        print now
        print self.public_opening_timestamp
        return now > self.public_opening_timestamp\
            and now < self.closing_timestamp


# many-to-many
class EventScoreType(db.Model):
    __tablename__ = 'event_score_types'

    id = db.Column(db.Integer, primary_key=True)
    # Can be renamed, default is the same as score_type.name
    name = db.Column(db.String, nullable=False)

    event_id = db.Column(
        db.Integer, db.ForeignKey('events.id'), nullable=False)
    event = db.relationship(
        'Event', backref=db.backref('score_types'))

    score_type_id = db.Column(
        db.Integer, db.ForeignKey('score_types.id'), nullable=False)
    score_type = db.relationship(
        'ScoreType', backref=db.backref('events'))

    # display precision in n_digits
    # default is the same as score_type.precision
    precision = db.Column(db.Integer)

    db.UniqueConstraint(event_id, score_type_id, name='es_constraint')

    def __init__(self, event, score_type, name=None, precision=None):
        self.event = event
        self.score_type = score_type
        if name is None:
            self.name = score_type.name
        if precision is None:
            self.precision = score_type.precision

    def __repr__(self):
        repr = '{}/{}'.format(self.event, self.team)
        return repr

    @property
    def score_function(self):
        return self.score_type.score_function


class CVFold(db.Model):
    """Storing train and test folds, more precisely: train and test indices.

    Created when the ramp event is set up.
    """

    __tablename__ = 'cv_folds'

    id = db.Column(db.Integer, primary_key=True)
    train_is = db.Column(NumpyType, nullable=False)
    test_is = db.Column(NumpyType, nullable=False)

    event_id = db.Column(
        db.Integer, db.ForeignKey('events.id'), nullable=False)
    event = db.relationship(
        'Event', backref=db.backref('cv_folds'))

    def __repr__(self):
        return 'fold {}'.format(self.train_is)[:15]


class EventAdmin(db.Model):
    __tablename__ = 'event_admins'

    id = db.Column(db.Integer, primary_key=True)

    event_id = db.Column(
        db.Integer, db.ForeignKey('events.id'), nullable=False)
    event = db.relationship(
        'Event', backref=db.backref('event_admins'))

    admin_id = db.Column(
        db.Integer, db.ForeignKey('users.id'), nullable=False)
    admin = db.relationship(
        'User', backref=db.backref('admined_events'))


# many-to-many
class EventTeam(db.Model):
    __tablename__ = 'event_teams'

    id = db.Column(db.Integer, primary_key=True)

    event_id = db.Column(
        db.Integer, db.ForeignKey('events.id'), nullable=False)
    event = db.relationship(
        'Event', backref=db.backref('event_teams'))

    team_id = db.Column(
        db.Integer, db.ForeignKey('teams.id'), nullable=False)
    team = db.relationship(
        'Team', backref=db.backref('team_events'))

    is_active = db.Column(db.Boolean, default=True)
    last_submission_name = db.Column(db.String, default=None)
    signup_timestamp = db.Column(db.DateTime, nullable=False)

    db.UniqueConstraint(event_id, team_id, name='et_constraint')

    def __init__(self, event, team):
        self.event = event
        self.team = team
        self.signup_timestamp = datetime.datetime.utcnow()

    def __repr__(self):
        repr = '{}/{}'.format(self.event, self.team)
        return repr


def get_active_user_event_team(event, user):
    # There should always be an active user team, if not, throw an exception
    event_teams = EventTeam.query.filter_by(event=event).all()
    for event_team in event_teams:
        if user in get_team_members(event_team.team) and event_team.is_active:
            return event_team


class SubmissionFileType(db.Model):
    __tablename__ = 'submission_file_types'

    id = db.Column(db.Integer, primary_key=True)
    # eg. 'code', 'text', 'data'
    name = db.Column(db.String, nullable=False, unique=True)
    is_editable = db.Column(db.Boolean, default=True)
    max_size = db.Column(db.Integer, default=None)


class Extension(db.Model):
    __tablename__ = 'extensions'

    id = db.Column(db.Integer, primary_key=True)
    # eg. 'py', 'csv', 'R'
    name = db.Column(db.String, nullable=False, unique=True)


# many-to-many connection between SubmissionFileType and Extension
class SubmissionFileTypeExtension(db.Model):
    __tablename__ = 'submission_file_type_extensions'

    id = db.Column(db.Integer, primary_key=True)

    type_id = db.Column(
        db.Integer, db.ForeignKey('submission_file_types.id'), nullable=False)
    type = db.relationship(
        'SubmissionFileType', backref=db.backref('extensions'))

    extension_id = db.Column(
        db.Integer, db.ForeignKey('extensions.id'), nullable=False)
    extension = db.relationship(
        'Extension', backref=db.backref('submission_file_types'))

    db.UniqueConstraint(type_id, extension_id, name='we_constraint')

    @property
    def file_type(self):
        return self.type.name

    @property
    def extension_name(self):
        return self.extension.name


class WorkflowElementType(db.Model):
    __tablename__ = 'workflow_element_types'

    id = db.Column(db.Integer, primary_key=True)
    # file name without extension
    # eg, regressor, classifier, external_data
    name = db.Column(db.String, nullable=False, unique=True)

    # eg, code, text, data
    type_id = db.Column(
        db.Integer, db.ForeignKey('submission_file_types.id'), nullable=False)
    type = db.relationship(
        'SubmissionFileType', backref=db.backref('workflow_element_types'))

    def __repr__(self):
        repr = 'WorkflowElementType(name={}, type={}, is_editable={}, max_size={})'.format(
            self.name, self.type.name, self.type.is_editable,
            self.type.max_size)
        return repr

    @property
    def file_type(self):
        return self.type.name

    @property
    def is_editable(self):
        return self.type.is_editable

    @property
    def max_size(self):
        return self.type.max_size


# training and test code now belongs to the workflow, not the workflow
# element. This latter would requre to carefully define workflow element
# interfaces. Eg, a dilemma: classifier + calibrator needs to handled at the
# workflow level (since calibrator needs held out data). Eventually we should
# have both workflow-level and workflow-element-level code to avoid code
# repetiotion.
class Workflow(db.Model):
    __tablename__ = 'workflows'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False, unique=True)

    def __repr__(self):
        repr = 'Workflow({})'.format(self.name)
        for workflow_element in self.elements:
            repr += '\n\t' + str(workflow_element)
        return repr

    @property
    def module(self):
        return import_module('.' + self.name, config.workflows_module)

    @property
    def train_submission(self):
        return self.module.train_submission

    @property
    def test_submission(self):
        return self.module.test_submission


# In lists we will order files according to their ids
# many-to-many link
# For now files define the workflow, so eg, a feature_extractor + regressor
# is not the same workflow as a feature_extractor + regressor + external data,
# even though the training codes are the same.
class WorkflowElement(db.Model):
    __tablename__ = 'workflow_elements'

    id = db.Column(db.Integer, primary_key=True)
    # Normally name will be the same as workflow_element_type.type.name,
    # unless specified otherwise. It's because in more complex workflows
    # the same type can occur more then once. self.type below will always
    # refer to workflow_element_type.type.name
    name = db.Column(db.String, nullable=False, unique=True)

    workflow_id = db.Column(
        db.Integer, db.ForeignKey('workflows.id'))
    workflow = db.relationship(
        'Workflow', backref=db.backref('elements'))

    workflow_element_type_id = db.Column(
        db.Integer, db.ForeignKey('workflow_element_types.id'),
        nullable=False)
    workflow_element_type = db.relationship(
        'WorkflowElementType', backref=db.backref('submission_files'))

    def __init__(self, workflow, workflow_element_type, name_in_workflow=None):
        self.workflow = workflow
        self.workflow_element_type = workflow_element_type
        if name_in_workflow is None:
            self.name = self.workflow_element_type.name
        else:
            self.name = name_in_workflow

    def __repr__(self):
        return 'WorkflowElement({})'.format(self.name)

    # e.g. 'regression', 'external_data'. Normally == name
    @property
    def type(self):
        return self.workflow_element_type.name

    @property
    def file_type(self):
        return self.workflow_element_type.file_type

    @property
    def is_editable(self):
        return self.workflow_element_type.is_editable

    @property
    def max_size(self):
        return self.workflow_element_type.max_size


# TODO: we should have a SubmissionWorkflowElementType table, describing the
# type of files we are expecting for a given RAMP. Fast unit test should be
# set up there, and each file should be unit tested right after submission.
# Kozmetics: erhaps mark which file the leaderboard link should point to (right
# now it is set to the first file in the list which is arbitrary).
# We will also have to handle auxiliary files (like csvs or other classes).
# User interface could have a sinlge submission form with a menu containing
# the file names for a given ramp + an "other" field when users will have to
# name their files
class SubmissionFile(db.Model):
    __tablename__ = 'submission_files'

    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.id'), nullable=False)
    submission = db.relationship(
        'Submission',
        backref=db.backref('files', cascade="all, delete-orphan"))

    # e.g. 'regression', 'external_data'
    workflow_element_id = db.Column(
        db.Integer, db.ForeignKey('workflow_elements.id'),
        nullable=False)
    workflow_element = db.relationship(
        'WorkflowElement', backref=db.backref('submission_files'))

    # e.g., ('code', 'py'), ('data', 'csv')
    submission_file_type_extension_id = db.Column(
        db.Integer, db.ForeignKey('submission_file_type_extensions.id'),
        nullable=False)
    submission_file_type_extension = db.relationship(
        'SubmissionFileTypeExtension', backref=db.backref('submission_files'))

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
        return '<a href="' + self.link + '">' + self.name[:20] + '</a>'

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


def combine_predictions_list(predictions_list, index_list=None):
    """Combine predictions in predictions_list[index_list].

    By taking the mean of their get_combineable_predictions views.

    E.g. for regression it is the actual
    predictions, and for classification it is the probability array (which
    should be calibrated if we want the best performance). Called both for
    combining one submission on cv folds (a single model that is trained on
    different folds) and several models on a single fold.
    Called by
    _get_bagging_score : which combines bags of the same model, trained on
        different folds, on the heldout test set
    _get_cv_bagging_score : which combines cv-bags of the same model, trained
        on different folds, on the training set
    get_next_best_single_fold : which does one step of the greedy forward
        selection (of different models) on a single fold
    _get_combined_predictions_single_fold : which does the full loop of greedy
        forward selection (of different models), until improvement, on a single
        fold
    _get_combined_test_predictions_single_fold : which computes the combination
        (constructed on the cv valid set) on the holdout test set, on a single
        fold
    _get_combined_test_predictions : which combines the foldwise combined
        and foldwise best test predictions into a single megacombination

    Parameters
    ----------
    predictions_list : list of instances of Predictions
        Each element of the list is an instance of Predictions of a given model
        on the same data points.
    index_list : None | list of integers
        The subset of predictions to be combined. If None, the full set is
        combined.

    Returns
    -------
    combined_predictions : instance of Predictions
        A predictions instance containing the combined (averaged) predictions.
    """
    if index_list is None:  # we combine the full list
        index_list = range(len(predictions_list))

    y_comb_list = np.array(
        [predictions_list[i].y_pred_comb for i in index_list])

    Predictions = type(predictions_list[0])

    y_comb = np.nanmean(y_comb_list, axis=0)
    combined_predictions = Predictions(y_pred=y_comb)
    return combined_predictions


def _get_score_cv_bags(event, score_type, predictions_list, true_predictions,
                       test_is_list=None):
    """
    Computed the bagged score of the predictions in predictions_list.

    Called by Submission.compute_valid_score_cv_bag and
    db_tools.compute_contributivity.

    Parameters
    ----------
    event : instance of Event
        Needed for the type of y_comb and
    predictions_list : list of instances of Predictions
    true_predictions : instance of Predictions
    test_is_list : list of integers
        Indices of points that should be bagged in each prediction. If None,
        the full prediction vectors will be bagged.
    Returns
    -------
    score_cv_bags : instance of Score ()
    """
    if test_is_list is None:  # we combine the full list
        test_is_list = [range(len(predictions.y_pred))
                        for predictions in predictions_list]

    n_samples = true_predictions.n_samples
    y_comb = np.array(
        [event.prediction.Predictions(n_samples=n_samples)
         for _ in predictions_list])
    score_cv_bags = []
    for i, test_is in enumerate(test_is_list):
        y_comb[i].set_valid_in_train(predictions_list[i], test_is)
        combined_predictions = combine_predictions_list(y_comb[:i + 1])
        valid_indexes = combined_predictions.valid_indexes
        score_cv_bags.append(score_type.score_function(
            true_predictions, combined_predictions, valid_indexes))
        # XXX maybe use masked arrays rather than passing valid_indexes
    return score_cv_bags


class SubmissionScore(db.Model):
    __tablename__ = 'submission_scores'

    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.id'), nullable=False)
    submission = db.relationship('Submission', backref=db.backref('scores'))

    score_type_id = db.Column(
        db.Integer, db.ForeignKey('score_types.id'), nullable=False)
    score_type = db.relationship(
        'ScoreType', backref=db.backref('submissions'))

    # These are cv-bagged scores. Individual scores are found in
    # SubmissionToTrain
    valid_score_cv_bag = db.Column(db.Float)  # cv
    test_score_cv_bag = db.Column(db.Float)  # holdout
    # we store the partial scores so to see the saturation and
    # overfitting as the number of cv folds grow
    valid_score_cv_bags = db.Column(NumpyType)
    test_score_cv_bags = db.Column(NumpyType)

    @property
    def score_name(self):
        return self.score_type.name

    @property
    def score_function(self):
        return self.score_type.score_function

    # default display precision in n_digits
    @property
    def precision(self):
        return self.score_type.precision

    @property
    def train_score_cv_mean(self):
        return np.array([ts.train_score for ts in self.on_cv_folds]).mean()

    @property
    def valid_score_cv_mean(self):
        return np.array([ts.valid_score for ts in self.on_cv_folds]).mean()

    @property
    def test_score_cv_mean(self):
        return np.array([ts.test_score for ts in self.on_cv_folds]).mean()

    @property
    def train_score_cv_std(self):
        return np.array([ts.train_score for ts in self.on_cv_folds]).std()

    @property
    def valid_score_cv_std(self):
        return np.array([ts.valid_score for ts in self.on_cv_folds]).std()

    @property
    def test_score_cv_std(self):
        return np.array([ts.test_score for ts in self.on_cv_folds]).std()


# evaluate right after train/test, so no need for 'scored' states
submission_states = db.Enum(
    'new', 'checked', 'checking_error', 'trained', 'training_error',
    'validated', 'validating_error', 'tested', 'testing_error')


class Submission(db.Model):
    """An abstract (untrained) submission."""

    __tablename__ = 'submissions'

    id = db.Column(db.Integer, primary_key=True)

    event_team_id = db.Column(
        db.Integer, db.ForeignKey('event_teams.id'), nullable=False)
    event_team = db.relationship(
        'EventTeam', backref=db.backref('submissions'))

    name = db.Column(db.String(20, convert_unicode=True), nullable=False)
    hash_ = db.Column(db.String, nullable=False, index=True, unique=True)
    submission_timestamp = db.Column(db.DateTime, nullable=False)
    training_timestamp = db.Column(db.DateTime)

    contributivity = db.Column(db.Float, default=0.0)

    state = db.Column(submission_states, default='new')
    # TODO: hide absolute path in error
    error_msg = db.Column(db.String, default='')
    # user can delete but we keep
    is_valid = db.Column(db.Boolean, default=True)
    # We can forget bad models.
    # If false, don't combine and set contributivity to zero
    is_to_ensemble = db.Column(db.Boolean, default=True)
    notes = db.Column(db.String, default='')  # eg, why is it disqualified

    # later also ramp_id
    db.UniqueConstraint(event_team_id, name, name='ts_constraint')

    def __init__(self, name, event_team):
        self.name = name
        self.event_team = event_team
        sha_hasher = hashlib.sha1()
        sha_hasher.update(self.event.name.encode('utf-8'))
        sha_hasher.update(self.team.name.encode('utf-8'))
        sha_hasher.update(self.name.encode('utf-8'))
        # We considered using the id, but then it will be given away in the
        # url which is maybe not a good idea.
        self.hash_ = '{}'.format(sha_hasher.hexdigest())
        self.submission_timestamp = datetime.datetime.utcnow()
        event_score_types = EventScoreType.query.filter_by(
            event=event_team.event)
        for event_score_type in event_score_types:
            submission_score = SubmissionScore(
                submission=self, score_type=event_score_type.score_type)
            db.session.add(submission_score)
        self.reset()
        print self

    def __str__(self):
        return 'Submission({}/{}/{})'.format(
            self.event.name, self.team.name, self.name)

    def __repr__(self):
        repr = '''Submission(event_name={}, team_name={}, name={}, files={},
                  state={}, train_time={})'''.format(
            self.event.name, self.team.name, self.name, self.files,
            self.state, self.train_time_cv_mean)
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
    def prediction(self):
        return self.event.prediction

    @hybrid_property
    def is_not_sandbox(self):
        return self.name != config.sandbox_d_name

    @hybrid_property
    def is_error(self):
        return (self.state == 'training_error') |\
            (self.state == 'checking_error') |\
            (self.state == 'validating_error') |\
            (self.state == 'testing_error')

    @hybrid_property
    def is_public_leaderboard(self):
        return self.is_not_sandbox & self.is_valid & (
            (self.state == 'validated') |
            (self.state == 'tested'))

    @hybrid_property
    def is_private_leaderboard(self):
        return self.is_not_sandbox & self.is_valid & (self.state == 'tested')

    @property
    def path(self):
        return os.path.join(
            config.submissions_path, 'submission_' + '{0:09d}'.format(self.id))

    @property
    def module(self):
        return self.path.lstrip('./').replace('/', '.')

    @property
    def f_names(self):
        return [file.f_name for file in self.files]

    @property
    def name_with_link(self):
        return '<a href="' + self.files[0].link + '">' + self.name[:20] +\
            '</a>'

    @property
    def state_with_link(self):
        return '<a href=/' + os.path.join(self.hash_, 'error.txt')\
            + '>' + self.state + '</a>'

    @property
    def train_time_cv_mean(self):
        return np.array([ts.train_time for ts in self.on_cv_folds]).mean()

    @property
    def valid_time_cv_mean(self):
        return np.array([ts.valid_time for ts in self.on_cv_folds]).mean()

    @property
    def test_time_cv_mean(self):
        return np.array([ts.test_time for ts in self.on_cv_folds]).mean()

    @property
    def train_time_cv_std(self):
        return np.array([ts.train_time for ts in self.on_cv_folds]).std()

    @property
    def valid_time_cv_std(self):
        return np.array([ts.valid_time for ts in self.on_cv_folds]).std()

    @property
    def test_time_cv_std(self):
        return np.array([ts.test_time for ts in self.on_cv_folds]).std()

    def set_state(self, state):
        self.state = state
        for submission_on_cv_fold in self.on_cv_folds:
            submission_on_cv_fold.state = state

    def reset(self):
        self.contributivity = 0.0
        self.state = 'new'
        self.error_msg = ''
        for score in self.scores:
            score.valid_score_cv_bag = score.score_type.worst
            score.test_score_cv_bag = score.score_type.worst
            score.valid_score_cv_bags = None
            score.test_score_cv_bags = None

    def set_error(self, error, error_msg):
        self.reset()
        self.state = error
        self.error_msg = error_msg
        for submission_on_cv_fold in self.on_cv_folds:
            submission_on_cv_fold.set_error(error, error_msg)

    def compute_valid_score_cv_bag(self):
        """Cv-bag cv_fold.valid_predictions using combine_predictions_list.

        The predictions in predictions_list[i] belong to those indicated
        by self.on_cv_folds[i].test_is.
        """
        true_predictions_train = self.event.problem.true_predictions_train()

        if self.is_public_leaderboard:
            predictions_list = [submission_on_cv_fold.valid_predictions for
                                submission_on_cv_fold in self.on_cv_folds]
            test_is_list = [submission_on_cv_fold.cv_fold.test_is for
                            submission_on_cv_fold in self.on_cv_folds]
            for score in self.scores:
                score.valid_score_cv_bags = _get_score_cv_bags(
                    self.event, score.score_type, predictions_list,
                    true_predictions_train, test_is_list)
                score.valid_score_cv_bag = float(score.valid_score_cv_bags[-1])
        else:
            for score in self.scores:
                score.valid_score_cv_bag = float(score.score_type.worst)
                score.valid_score_cv_bags = None
        db.session.commit()

    def compute_test_score_cv_bag(self):
        """Bag cv_fold.test_predictions using combine_predictions_list.

        And stores the score of the bagged predictor in test_score_cv_bag. The
        scores of partial combinations are stored in test_score_cv_bags.
        This is for assessing the bagging learning curve, which is useful for
        setting the number of cv folds to its optimal value (in case the RAMP
        is competitive, say, to win a Kaggle challenge; although it's kinda
        stupid since in those RAMPs we don't have a test file, so the learning
        curves should be assessed in compute_valid_score_cv_bag on the
        (cross-)validation sets).
        """
        if self.is_private_leaderboard:
            # When we have submission id in Predictions, we should get the
            # team and submission from the db
            true_predictions = self.event.problem.true_predictions_test()
            predictions_list = [submission_on_cv_fold.test_predictions for
                                submission_on_cv_fold in self.on_cv_folds]
            combined_predictions_list = [
                combine_predictions_list(predictions_list[:i + 1]) for
                i in range(len(predictions_list))]
            for score in self.scores:
                score.test_score_cv_bags = [
                    score.score_function(
                        true_predictions, combined_predictions) for
                    combined_predictions in combined_predictions_list]
                score.test_score_cv_bag = float(score.test_score_cv_bags[-1])
        else:
            for score in self.scores:
                score.test_score_cv_bag = float(score.score_type.worst)
                score.test_score_cv_bags = None
        db.session.commit()

    # contributivity could be a property but then we could not query on it
    def set_contributivity(self, is_commit=True):
        self.contributivity = 0.0
        if self.is_public_leaderboard:
            # we share a unit of 1. among folds
            unit_contributivity = 1. / len(self.on_cv_folds)
            for submission_on_cv_fold in self.on_cv_folds:
                self.contributivity +=\
                    unit_contributivity * submission_on_cv_fold.contributivity
        if is_commit:
            db.session.commit()

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


def get_next_best_single_fold(event, predictions_list, true_predictions,
                              best_index_list):
    """.

    Find the model that minimizes the score if added to
    predictions_list[best_index_list] using event.official_score_function.
    If there is no model improving the input
    combination, the input best_index_list is returned. Otherwise the best
    model is added to the list. We could also return the combined prediction
    (for efficiency, so the combination would not have to be done each time;
    right now the algo is quadratic), but I don't think any meaningful
    rule will be associative, in which case we should redo the combination from
    scratch each time the set changes. Since now combination = mean, we could
    maintain the sum and the number of models, but it would be a bit bulky.
    We'll see how this evolves.

    Parameters
    ----------
    predictions_list : list of instances of Predictions
        Each element of the list is an instance of Predictions of a model
        on the same (cross-validation valid) data points.
    true_predictions : instance of Predictions
        The ground truth.
    best_index_list : list of integers
        Indices of the current best model.

    Returns
    -------
    best_index_list : list of integers
        Indices of the models in the new combination. If the same as input,
        no models wer found improving the score.
    """
    best_predictions = combine_predictions_list(
        predictions_list, index_list=best_index_list)
    best_score = event.official_score_function(
        true_predictions, best_predictions)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a
    # model is added several times, it's upweighted, leading to
    # integer-weighted ensembles
    r = np.arange(len(predictions_list))
    # Randomization doesn't matter, only in case of exact equality.
    # np.random.shuffle(r)
    # print r
    for i in r:
        combined_predictions = combine_predictions_list(
            predictions_list, index_list=np.append(best_index_list, i))
        new_score = event.official_score_function(
            true_predictions, combined_predictions)
        is_lower_the_better = event.official_score_type.is_lower_the_better
        if (is_lower_the_better and new_score < best_score) or\
                (not is_lower_the_better and new_score > best_score):
            best_predictions = combined_predictions
            best_index = i
            best_score = new_score
    if best_index > -1:
        return np.append(best_index_list, best_index), best_score
    else:
        return best_index_list, best_score


class SubmissionScoreOnCVFold(db.Model):
    __tablename__ = 'submission_score_on_cv_folds'

    id = db.Column(db.Integer, primary_key=True)
    submission_on_cv_fold_id = db.Column(
        db.Integer, db.ForeignKey('submission_on_cv_folds.id'), nullable=False)
    submission_on_cv_fold = db.relationship(
        'SubmissionOnCVFold', backref=db.backref('scores'))

    submission_score_id = db.Column(
        db.Integer, db.ForeignKey('submission_scores.id'), nullable=False)
    submission_score = db.relationship(
        'SubmissionScore', backref=db.backref('on_cv_folds'))

    train_score = db.Column(db.Float)
    valid_score = db.Column(db.Float)
    test_score = db.Column(db.Float)

    db.UniqueConstraint(
        submission_on_cv_fold_id, submission_score_id, name='ss_constraint')

    @property
    def score_type(self):
        return self.submission_score.score_type

    @property
    def score_function(self):
        return self.score_type.score_function

# TODO: rename submission to workflow and submitted file to workflow_element
# TODO: SubmissionOnCVFold should actually be a workflow element. Saving
# train_pred means that we can input it to the next workflow element
# TODO: implement check
class SubmissionOnCVFold(db.Model):
    """SubmissionOnCVFold.

    is an instantiation of Submission, to be trained on a data file and a cv
    fold. We don't actually store the trained model in the db (lack of disk and
    pickling issues), so trained submission is not a database column. On the
    other hand, we will store train, valid, and test predictions. In a sense
    substituting CPU time for storage.
    """

    __tablename__ = 'submission_on_cv_folds'

    id = db.Column(db.Integer, primary_key=True)

    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.id'), nullable=False)
    submission = db.relationship(
        'Submission', backref=db.backref(
            'on_cv_folds', cascade="all, delete-orphan"))

    cv_fold_id = db.Column(
        db.Integer, db.ForeignKey('cv_folds.id'), nullable=False)
    cv_fold = db.relationship(
        'CVFold', backref=db.backref(
            'submissions', cascade="all, delete-orphan"))

    # filled by cv_fold.get_combined_predictions
    contributivity = db.Column(db.Float, default=0.0)
    best = db.Column(db.Boolean, default=False)

    # prediction on the full training set, including train and valid points
    # properties train_predictions and valid_predictions will make the slicing
    full_train_y_pred = db.Column(NumpyType, default=None)
    test_y_pred = db.Column(NumpyType, default=None)
    train_time = db.Column(db.Float, default=0.0)
    valid_time = db.Column(db.Float, default=0.0)
    test_time = db.Column(db.Float, default=0.0)
    state = db.Column(submission_states, default='new')
    error_msg = db.Column(db.String, default='')

    db.UniqueConstraint(submission_id, cv_fold_id, name='sc_constraint')

    def __init__(self, submission, cv_fold):
        self.submission = submission
        self.cv_fold = cv_fold
        for score in submission.scores:
            submission_score_on_cv_fold = SubmissionScoreOnCVFold(
                submission_on_cv_fold=self, submission_score=score)
            db.session.add(submission_score_on_cv_fold)
        self.reset()

    def __repr__(self):
        repr = 'state = {}, c = {}'\
            ', best = {}'.format(
                self.state, self.contributivity, self.best)
        return repr

    @hybrid_property
    def is_public_leaderboard(self):
        return (self.state == 'validated') | (self.state == 'tested')

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
        return self.submission.prediction.Predictions(
            y_pred=self.full_train_y_pred)

    @property
    def train_predictions(self):
        return self.submission.prediction.Predictions(
            y_pred=self.full_train_y_pred[self.cv_fold.train_is])

    @property
    def valid_predictions(self):
        return self.submission.prediction.Predictions(
            y_pred=self.full_train_y_pred[self.cv_fold.test_is])

    @property
    def test_predictions(self):
        return self.submission.prediction.Predictions(y_pred=self.test_y_pred)

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
            score.train_score = score.score_type.worst
            score.valid_score = score.score_type.worst
            score.test_score = score.score_type.worst

    def set_error(self, error, error_msg):
        self.reset()
        self.state = error
        self.error_msg = error_msg

    def compute_train_scores(self):
        true_full_train_predictions =\
            self.submission.event.problem.true_predictions_train()
        for score in self.scores:
            score.train_score = float(score.score_function(
                true_full_train_predictions, self.full_train_predictions,
                self.cv_fold.train_is))
        db.session.commit()

    def compute_valid_scores(self):
        true_full_train_predictions =\
            self.submission.event.problem.true_predictions_train()
        for score in self.scores:
            score.valid_score = float(score.score_function(
                true_full_train_predictions, self.full_train_predictions,
                self.cv_fold.test_is))
        db.session.commit()

    def compute_test_scores(self):
        true_test_predictions =\
            self.submission.event.problem.true_predictions_test()
        for score in self.scores:
            score.test_score = float(score.score_function(
                true_test_predictions, self.test_predictions))
        db.session.commit()

    def update(self, detached_submission_on_cv_fold):
        """From trained DetachedSubmissionOnCVFold."""
        self.state = detached_submission_on_cv_fold.state
        if self.is_error:
            self.error_msg = detached_submission_on_cv_fold.error_msg
        else:
            if self.state in ['trained', 'validated', 'tested']:
                self.train_time = detached_submission_on_cv_fold.train_time
            if self.state in ['validated', 'tested']:
                self.valid_time = detached_submission_on_cv_fold.valid_time
                self.full_train_y_pred =\
                    detached_submission_on_cv_fold.full_train_y_pred
                self.compute_train_scores()
                self.compute_valid_scores()
            if self.state in ['tested']:
                self.test_time = detached_submission_on_cv_fold.test_time
                self.test_y_pred = detached_submission_on_cv_fold.test_y_pred
                self.compute_test_scores()
        db.session.commit()


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
        self.module = submission_on_cv_fold.submission.module
        self.error_msg = submission_on_cv_fold.error_msg
        self.train_time = submission_on_cv_fold.train_time
        self.valid_time = submission_on_cv_fold.valid_time
        self.test_time = submission_on_cv_fold.test_time
        self.trained_submission = None
        self.train_submission =\
            submission_on_cv_fold.submission.event.train_submission
        self.test_submission =\
            submission_on_cv_fold.submission.event.test_submission

    def __repr__(self):
        repr = 'Submission({}) on fold {}'.format(
            self.name, str(self.train_is)[:10])
        return repr


class UserInteraction(db.Model):
    __tablename__ = 'user_interactions'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    interaction = db.Column(db.String, nullable=False)
    note = db.Column(db.String, default=None)
    submission_file_diff = db.Column(db.String, default=None)
    submission_file_similarity = db.Column(db.Float, default=None)
    ip = db.Column(db.String, default=None)

    user_id = db.Column(
        db.Integer, db.ForeignKey('users.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('user_interactions'))

    event_team_id = db.Column(
        db.Integer, db.ForeignKey('event_teams.id'))
    event_team = db.relationship(
        'EventTeam', backref=db.backref('user_interactions'))

    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.id'))
    submission = db.relationship(
        'Submission', backref=db.backref('user_interactions'))

    submission_file_id = db.Column(
        db.Integer, db.ForeignKey('submission_files.id'))
    submission_file = db.relationship(
        'SubmissionFile', backref=db.backref('user_interactions'))

    def __init__(self, user, interaction, event=None, note=None,
                 submission=None, submission_file=None, diff=None,
                 similarity=None):
        self.timestamp = datetime.datetime.utcnow()
        self.interaction = interaction
        self.user = user
        if event is not None:
            self.event_team = get_active_user_event_team(event, user)
        self.ip = request.environ['REMOTE_ADDR']
        self.note = note
        self.submission = submission
        self.submission_file = submission_file
        self.submission_file_diff = diff
        self.submission_file_similarity = similarity

    @property
    def submission_file_diff_link(self):
        if self.submission_file_diff is None:
            return None
        return os.path.join(
            config.submissions_path, 'diff_bef24208a45043059', str(self.id))


class NameClashError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MergeTeamError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DuplicateSubmissionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TooEarlySubmissionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MissingSubmissionFileError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MissingExtensionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
