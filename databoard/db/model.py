import os
import zlib
import string
import timeit
import logging
import hashlib
import datetime
import numpy as np
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
import databoard.config as config
import databoard.generic as generic

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + config.db_f_name
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
db = SQLAlchemy(app)

max_members_per_team = 3  # except for users own team
opening_timestamp = None
public_opening_timestamp = None  # before teams can see only their own scores
closing_timestamp = None

logger = logging.getLogger('databoard')

# Training set table
# Problem table
# Ramp table that connects all (training set, problem, cv, specific)
# specific should be cut at least into problem-specific, data and cv-specific,
# and ramp (event) specific files


class NumpyType(db.TypeDecorator):
    """ Storing zipped numpy arrays."""
    impl = db.LargeBinary

    def process_bind_param(self, value, dialect):
        # we convert the initial value into np.array to handle None and lists
        return zlib.compress(np.array(value).dumps())

    def process_result_value(self, value, dialect):
        return np.loads(zlib.decompress(value))


class ScoreType(db.TypeDecorator):
    """ Storing score types (with redefined comparators)."""
    impl = db.Float

    def process_bind_param(self, value, dialect):
        if type(value) == float:
            return value
        else:
            specific = config.config_object.specific
            return specific.score.revert(value)

    def process_result_value(self, value, dialect):
        specific = config.config_object.specific
        return specific.score.convert(value)


class User(db.Model):
    __tablename__ = 'users'

    id_ = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)
    hashed_password = db.Column(db.String, nullable=False)
    lastname = db.Column(db.String, nullable=False)
    firstname = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False, unique=True)
    linkedin_url = db.Column(db.String, default=None)
    twitter_url = db.Column(db.String, default=None)
    facebook_url = db.Column(db.String, default=None)
    google_url = db.Column(db.String, default=None)
    access_level = db.Column(db.Enum('admin', 'user'), default='user')
    signup_timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    is_validated = db.Column(db.Boolean, default=False)  # admin has to valid

    def __str__(self):
        str_ = 'User({}, admined=['.format(self.name)
        str_ += string.join([team.name for team in self.admined_teams], ', ')
        str_ += '])'
        return str_

    def __repr__(self):
        repr = '''User(name={}, lastname={}, firstname={}, email={},
                  admined_teams={})'''.format(
            self.name, self.lastname, self.firstname, self.email,
            self.admined_teams)
        return repr


class Team(db.Model):
    __tablename__ = 'teams'

    id_ = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)

    admin_id = db.Column(db.Integer, db.ForeignKey('users.id_'))
    admin = db.relationship('User', backref=db.backref('admined_teams'))

    # initiator asks for merge, acceptor accepts
    initiator_id = db.Column(
        db.Integer, db.ForeignKey('teams.id_'), default=None)
    initiator = db.relationship(
        'Team', primaryjoin=('Team.initiator_id == Team.id_'), uselist=False)

    acceptor_id = db.Column(
        db.Integer, db.ForeignKey('teams.id_'), default=None)
    acceptor = db.relationship(
        'Team', primaryjoin=('Team.acceptor_id == Team.id_'), uselist=False)

    creation_timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)  # ->ramp_teams

    def __str__(self):
        str_ = 'Team({}, admin={})'.format(self.name, self.admin.name)
        return str_

    def __repr__(self):
        repr = '''Team(name={}, admin_name={}, is_active={},
                  initiator={}, acceptor={})'''.format(
            self.name, self.admin.name, self.is_active, self.initiator,
            self.acceptor)
        return repr


# TODO: we shuold have a SubmissionFileType table, describing the type
# of files we are expecting for a given RAMP. Fast unit test should be set up
# there, and each file should be unit tested right after submission.
# We should have a max_size attribute that we could set when setting up ramps.
# Kozmetics: erhaps mark which file the leaderboard link should point to (right
# now it is set to the first file in the list which is arbitrary).
# We will also have to handle auxiliary files (like csvs or other classes).
# User interface could have a sinlge submission form with a menu containing
# the file names for a given ramp + an "other" field when users will have to
# name their files
class SubmissionFile(db.Model):
    __tablename__ = 'submission_files'

    id_ = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.id_'), nullable=False)
    submission = db.relationship('Submission', backref=db.backref('files'))
    name = db.Column(db.String, nullable=False)

    db.UniqueConstraint(submission_id, name)

    def __init__(self, name, submission):
        self.name = name
        self.submission = submission
        if not os.path.isfile(self.path):
            raise MissingSubmissionFile('{}/{}/{}: {}'.format(
                submission.team.name, submission.name, name, self.path))

    @hybrid_property
    def path(self):
        return self.submission.path + os.path.sep + self.name

    def __repr__(self):
        return 'SubmissionFile(name={}, path={})'.format(
            self.name, self.path)


class Submission(db.Model):
    """An abstract (untrained) submission."""

    __tablename__ = 'submissions'

    id_ = db.Column(db.Integer, primary_key=True)

    team_id = db.Column(db.Integer, db.ForeignKey('teams.id_'), nullable=False)
    # one-to-many, ->ramp_teams
    team = db.relationship('Team', backref=db.backref('submissions'))

    name = db.Column(db.String(20), nullable=False)
    hash_ = db.Column(db.String, nullable=False)
    submission_timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow)
    training_timestamp = db.Column(db.DateTime)
    scoring_timestamp = db.Column(db.DateTime)

    # These are cv-bagged scores. Individual scores are found in
    # SubmissionToTrain
    valid_score_cv_bag = db.Column(ScoreType, default=0.0)  # cv
    test_score_cv_bag = db.Column(ScoreType, default=0.0)  # holdout
    contributivity = db.Column(db.Float, default=0.0)

    # evaluate right after train/test, so no need for 'scored' states
    state = db.Column(db.Enum(
        'new', 'checked', 'trained', 'tested',
        'train_scored', 'test_scored', 'checking_error', 'training_error',
        'testing_error', 'unit_testing_error', 'ignore'),
        default='new')
    is_valid = db.Column(
        db.Boolean, default=True)  # user can delete but we keep
    is_to_ensemble = db.Column(
        db.Boolean, default=True)  # we can forget bad models
    notes = db.Column(db.String, default='')  # eg, why is it disqualified

    db.UniqueConstraint(team_id, name)  # later also ramp_id

    def __init__(self, name, team):
        self.name = name
        self.team = team
        sha_hasher = hashlib.sha1()
        sha_hasher.update(self.team.name)
        sha_hasher.update(self.name)
        # We considered using the id, but then it will be given away in the
        # url which is maybe not a good idea.
        self.hash_ = 'm{}'.format(sha_hasher.hexdigest())

    @hybrid_property
    def is_public_leaderboard(self):
        return self.is_valid and self.state == 'train_scored'

    @property
    def path(self):
        path = os.path.join(
            config.submissions_path, self.team.name, self.hash_)
        return path

    @property
    def module(self):
        return self.path.lstrip('./').replace('/', '.')

    @property
    def f_names(self):
        return [file.name for file in self.files]

    @property
    def name_with_link(self):
        return '<a href="' + self.files[0].path + '">' + self.name + '</a>'

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
    def train_time_cv_mean(self):
        return np.array([ts.train_time for ts in self.on_cv_folds]).mean()

    @property
    def valid_time_cv_mean(self):
        return np.array([ts.valid_time for ts in self.on_cv_folds]).mean()

    @property
    def test_time_cv_mean(self):
        return np.array(
            [ts.test_time for ts in self.submission_on_cv_folds]).mean()

    def get_paths(self, submissions_path=config.submissions_path):
        team_path = os.path.join(submissions_path, self.team.name)
        submission_path = os.path.join(team_path, self.hash_)
        return team_path, submission_path

    def __str__(self):
        return 'Submission({}/{})'.format(self.team.name, self.name)

    def __repr__(self):
        repr = '''Submission(team_name={}, name={}, files={},
                  state={}, train_time={})'''.format(
            self.team.name, self.name, self.files,
            self.state, self.train_time_cv_mean)
        return repr

    def run_method_on_folds(self, after_state, error_state, doing, method, cv):
        import databoard.train_test as train_test

        logger.info('{} : {}/{}'.format(
            str.capitalize(doing), self.team.name, self.name))

        try:
            train_test.run_method_on_folds(self, method, cv)
            self.state = after_state
        except Exception, e:
            # TODO: better error handling, concrete methods should
            # set correct states, here we dont know whether train or test
            self.state = error_state
            if hasattr(e, 'traceback'):
                msg = str(e.traceback)
            else:
                msg = repr(e)
            logger.error('{} failed with exception: \n{}'.format(
                str.capitalize(doing), msg))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(generic.get_f_name(self.path, '.',
                                         error_state, 'txt'), 'w') as f:
                error_msg = msg
                cut_exception_text = error_msg.rfind('--->')
                if cut_exception_text > 0:
                    error_msg = error_msg[cut_exception_text:]
                f.write("{}".format(error_msg))


class CVFold(db.Model):
    """Created when the ramp is set up. Storing train and test folds, more
    precisely: train and tes indices. Shuold be related to the data set
    and the ramp (that defines the cv). """

    __tablename__ = 'cv_folds'

    id_ = db.Column(db.Integer, primary_key=True)
    train_is = db.Column(NumpyType, nullable=False)
    test_is = db.Column(NumpyType, nullable=False)

    def __repr__(self):
        repr = 'fold {}'.format(self.train_is)[:10]
        return repr


# TODO: rename submission to workflow and submitted file to workflow_element
# TODO: SubmissionOnCVFold should actually be a workflow element. Saving
# train_pred means that we can input it to the next workflow element
# TODO: implement check
class SubmissionOnCVFold(db.Model):
    """Submission is an abstract (untrained) submission. SubmissionOnCVFold
    is an instantiation of Submission, to be trained on a data file and a cv
    fold. We don't actually store the trained model in the db (lack of disk and
    pickling issues), so trained submission is not a database column. On the
    other hand, we will store train, valid, and test predictions. In a sense
    substituting CPU time for storage."""

    __tablename__ = 'submission_on_cv_folds'

    id_ = db.Column(db.Integer, primary_key=True)

    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.id_'), nullable=False)
    submission = db.relationship(
        'Submission', backref=db.backref('on_cv_folds'))

    cv_fold_id = db.Column(
        db.Integer, db.ForeignKey('cv_folds.id_'), nullable=False)
    cv_fold = db.relationship(
        'CVFold', backref=db.backref('submissions_on_cv_fold'))

    # prediction on the full training set, including train and valid points
    full_train_pred = db.Column(NumpyType, default=None)
    test_pred = db.Column(NumpyType, default=None)
    train_time = db.Column(db.Float, default=0.0)
    valid_time = db.Column(db.Float, default=0.0)
    test_time = db.Column(db.Float, default=0.0)
    train_score = db.Column(ScoreType, default=0.0)
    valid_score = db.Column(ScoreType, default=0.0)
    test_score = db.Column(ScoreType, default=0.0)
    state = db.Column(db.Enum(
        'new', 'checked', 'checking_error', 'trained', 'training_error',
        'validated', 'validating_error', 'tested', 'testing_error'),
        default='new')
    error_msg = db.Column(db.String, default='')

    # later also ramp_id or data_id
    db.UniqueConstraint(submission_id, cv_fold_id)

    def _make_error_message(self, e):
        """log_msg is the full error what we print into logger.error. error_msg
        is what we save and display to the user. Ideally error_msg is the part
        of the code that is related to the user submission.
        """
        if hasattr(e, 'traceback'):
            log_msg = str(e.traceback)
        else:
            log_msg = repr(e)
        error_msg = log_msg
        cut_exception_text = error_msg.rfind('--->')
        if cut_exception_text > 0:
            error_msg = error_msg[cut_exception_text:]
        return log_msg, error_msg

    def train(self, X, y, force_retrain=False):
        if self.state not in ['new', 'checked'] and not force_retrain:
            if 'error' in self.state:
                logger.error('Trying to train failed {} on {}'.format(
                    self.submission, self.cv_fold))
            else:
                logger.info('Already trained {} on {}'.format(
                    self.submission, self.cv_fold))
            return

        # so to make it importable, TODO: should go to make_submission
        open(os.path.join(self.submission.path, '__init__.py'), 'a').close()

        train_is = self.cv_fold.train_is
        test_is = self.cv_fold.test_is
        specific = config.config_object.specific

        logger.info('Training {} on {}'.format(
            self.submission, self.cv_fold))
        start = timeit.default_timer()
        try:
            trained_submission = specific.train_submission(
                self.submission.module, X, y, train_is)
            self.state = 'trained'
        except Exception, e:
            self.state = 'training_error'
            log_msg, self.error_msg = self._make_error_message(e)
            logger.error(
                'Training {} on {} failed with exception: \n{}'.format(
                    self.submission, self.cv_fold, log_msg))
            return
        end = timeit.default_timer()
        self.train_time = end - start

        logger.info('Validating {} on {}'.format(
            self.submission, self.cv_fold))
        start = timeit.default_timer()
        try:
            full_train_predictions = specific.test_submission(  # Predictions
                trained_submission, X, range(len(y)))
            self.state = 'validated'
        except Exception, e:
            self.state = 'validating_error'
            log_msg, self.error_msg = self._make_error_message(e)
            logger.error(
                'Validating {} on {} failed with exception: \n{}'.format(
                    self.submission, self.cv_fold, log_msg))
            return
        end = timeit.default_timer()
        self.valid_time = end - start

        self.full_train_pred = full_train_predictions.y_pred  # numpy array
        true_full_train_predictions = generic.get_true_predictions_train()
        self.train_score = specific.score(
            true_full_train_predictions, full_train_predictions, train_is)
        self.valid_score = specific.score(
            true_full_train_predictions, full_train_predictions, test_is)
        return trained_submission

    def test(self, X, y, trained_submission, force_retest=False):
        if self.state not in ['new', 'checked', 'trained', 'validated']\
                and not force_retest:
            if 'error' in self.state:
                logger.error('Trying to test failed {} on {}'.format(
                    self.submission, self.cv_fold))
            else:
                logger.info('Already tested {} on {}'.format(
                    self.submission, self.cv_fold))
            return

        specific = config.config_object.specific

        logger.info('Testing {} on {}'.format(
            self.submission, self.cv_fold))
        start = timeit.default_timer()
        try:
            test_predictions = specific.test_submission(  # Predictions type
                trained_submission, X, range(len(y)))
            self.state = 'tested'
        except Exception, e:
            self.state = 'testing_error'
            log_msg, self.error_msg = self._make_error_message(e)
            logger.error(
                'Testing {} on {} failed with exception: \n{}'.format(
                    self.submission, self.cv_fold, log_msg))
            return
        end = timeit.default_timer()
        self.test_time = end - start

        self.test_pred = test_predictions.y_pred  # numpy array
        true_test_predictions = generic.get_true_predictions_test()
        self.test_score = specific.score(
            true_test_predictions, test_predictions)

    def __repr__(self):
        repr = 'state = {}, valid_score = {}, test_score = {}'.format(
            self.state, self.valid_score, self.test_score)
        return repr

    @property
    def train_pred(self):
        return self.full_train_pred[self.cv_fold.train_is]

    @property
    def valid_pred(self):
        return self.full_train_pred[self.cv_fold.test_is]

    @property
    def train_predictions(self):
        specific = config.config_object.specific
        return specific.Predictions(y_pred=self.train_pred)

    @property
    def valid_predictions(self):
        specific = config.config_object.specific
        return specific.Predictions(y_pred=self.valid_pred)

    @property
    def test_predictions(self):
        specific = config.config_object.specific
        return specific.Predictions(self.test_pred)


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


class MissingSubmissionFile(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
