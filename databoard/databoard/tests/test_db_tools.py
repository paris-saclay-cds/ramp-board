import datetime
import os
import shutil

import pytest
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from rampwf.workflows import FeatureExtractorClassifier

from rampdb.model import CVFold
from rampdb.model import Event
from rampdb.model import EventScoreType
from rampdb.model import EventTeam
from rampdb.model import Submission
from rampdb.model import SubmissionOnCVFold
from rampdb.model import Problem
from rampdb.model import Team
from rampdb.model import User
from rampdb.model import Workflow

from rampdb.model import DuplicateSubmissionError
from rampdb.model import MissingExtensionError
from rampdb.model import MissingSubmissionFileError
from rampdb.model import NameClashError
from rampdb.model import TooEarlySubmissionError

from databoard import db
from databoard import deployment_path
from databoard import ramp_config

from databoard.testing import create_test_db
from databoard.testing import create_toy_db
from databoard.testing import _setup_ramp_kits_ramp_data

from databoard.utils import check_password

from databoard.db_tools import add_event
from databoard.db_tools import add_problem
from databoard.db_tools import add_workflow
from databoard.db_tools import approve_user
from databoard.db_tools import ask_sign_up_team
from databoard.db_tools import create_user
from databoard.db_tools import delete_event
from databoard.db_tools import delete_problem
from databoard.db_tools import get_team_members
from databoard.db_tools import get_new_submissions
from databoard.db_tools import get_submissions
from databoard.db_tools import get_submission_on_cv_folds
from databoard.db_tools import make_submission
from databoard.db_tools import make_submission_and_copy_files
from databoard.db_tools import sign_up_team
from databoard.db_tools import submit_starting_kit
from databoard.db_tools import update_submission_on_cv_fold
from databoard.db_tools import update_user

HERE = os.path.dirname(__file__)


class Bunch(dict):
    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


@pytest.fixture
def setup_db():
    try:
        create_test_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


@pytest.fixture(scope="module")
def setup_toy_db():
    try:
        create_toy_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def test_create_user(setup_db):
    name = 'test_user'
    password = 'test'
    lastname = 'Test'
    firstname = 'User'
    email = 'test.user@gmail.com'
    access_level = 'asked'
    create_user(name=name, password=password, lastname=lastname,
                firstname=firstname, email=email, access_level=access_level)
    users = db.session.query(User).all()
    assert len(users) == 1
    user = users[0]
    assert user.name == name
    assert check_password(password, user.hashed_password)
    assert user.lastname == lastname
    assert user.firstname == firstname
    assert user.email == email
    assert user.access_level == access_level
    # check that a team was automatically added with the new user
    team = db.session.query(Team).all()
    assert len(team) == 1
    team = team[0]
    assert team.name == name
    assert team.admin_id == user.id


def test_create_user_error_double(setup_db):
    create_user(name='test_user', password='test', lastname='Test',
                firstname='User', email='test.user@gmail.com',
                access_level='asked')
    # check that an error is raised when the username is already used
    err_msg = 'username is already in use and email is already in use'
    with pytest.raises(NameClashError, match=err_msg):
        create_user(name='test_user', password='test', lastname='Test',
                    firstname='User', email='test.user@gmail.com',
                    access_level='asked')
    # TODO: Check how to trigger the integrity error.


# TODO: We can only mock the FlaskForm for the moment
def test_update_user(setup_db):
    user = create_user(name='test_user', password='test', lastname='Test',
                       firstname='User', email='test.user@gmail.com',
                       access_level='asked')
    print(user.lastname)
    class Form:
        def __init__(self, lastname, firstname, linkedin_url, twitter_url,
                     facebook_url, google_url, github_url, website_url, bio,
                     email, is_want_news):
            self.lastname = Bunch(data=lastname)
            self.firstname = Bunch(data=firstname)
            self.linkedin_url = Bunch(data=linkedin_url)
            self.twitter_url = Bunch(data=twitter_url)
            self.facebook_url = Bunch(data=facebook_url)
            self.google_url = Bunch(data=google_url)
            self.github_url = Bunch(data=github_url)
            self.website_url = Bunch(data=website_url)
            self.bio = Bunch(data=bio)
            self.email = Bunch(data=email)
            self.is_want_news = Bunch(data=is_want_news)

    form = Form(lastname='new_lastname', firstname='new_firstname',
                linkedin_url='new_linkedin', twitter_url='new_twitter',
                facebook_url='new_facebook', google_url='new_google',
                github_url='new_github', website_url='new_website',
                bio='new bio', email='new.email@example.com',
                is_want_news=False)
    update_user(user, form)
    for attr in ['lastname', 'firstname', 'linkedin_url', 'twitter_url',
                 'facebook_url', 'google_url', 'github_url', 'website_url',
                 'bio', 'email', 'is_want_news']:
        assert getattr(user, attr) == getattr(form, attr).data


def test_approve_user(setup_db):
    create_user(name='test_user', password='test', lastname='Test',
                firstname='User', email='test.user@gmail.com',
                access_level='asked')
    user = db.session.query(User).all()[0]
    assert user.access_level == 'asked'
    approve_user(user.name)
    assert user.access_level == 'user'


def test_add_workflow(setup_db):
    ramp_workflow = FeatureExtractorClassifier()
    add_workflow(ramp_workflow)
    workflows = db.session.query(Workflow).all()
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.name == ramp_workflow.__class__.__name__
    for registered_elt, expected_elt in zip(workflow.elements,
                                            ramp_workflow.element_names):
        assert registered_elt.name == expected_elt
        registered_elt_type = registered_elt.workflow_element_type
        assert registered_elt_type.type.name == 'code'
        assert registered_elt_type.is_editable


class WorkflowFaultyElements:
    """Workflow with faulty elements."""
    def __init__(self, case=None):
        self.case = case

    @property
    def element_names(self):
        if self.case == 'multiple-dot':
            return ['too.much.dot.workflow']
        elif self.case == 'unknown-extension':
            return ['function.cpp']


@pytest.mark.parametrize(
    "case, err_msg",
    [('multiple-dot', 'should contain at most one "."'),
     ('unknown-extension', 'Unknown extension')]
)
def test_add_workflow_error(case, err_msg, setup_db):
    workflow = WorkflowFaultyElements(case=case)
    with pytest.raises(ValueError, match=err_msg):
        add_workflow(workflow)
    # TODO: there is no easy way to test a non valid type extension.


def _check_problem(name, workflow_name):
    """Check the problem in the database.

    Parameters
    ----------
    name : str
        Expected name of the problem.
    workflow_name : str
        The workflow name used in the problem.
    """
    problems = db.session.query(Problem).all()
    assert len(problems) == 1
    problem = problems[0]
    assert problem.name == name
    assert problem.workflow.name == workflow_name


def _setup_problem(problem_name):
    # a problem requires a name
    # setup the ramp-kit and ramp-data for the iris challenge
    _setup_ramp_kits_ramp_data(problem_name)
    add_problem(problem_name)


def test_add_problem(setup_db):
    problem_name = 'iris'
    # setup the ramp-kit and ramp-data for the iris challenge
    _setup_ramp_kits_ramp_data(problem_name)

    add_problem(problem_name)
    _check_problem(problem_name, 'Classifier')

    # Without forcing, we cannot write the same problem twice
    err_msg = 'Attempting to overwrite a problem and delete all linked events'
    with pytest.raises(ValueError, match=err_msg):
        add_problem(problem_name)

    # Force add the problem
    add_problem(problem_name, force=True)
    _check_problem(problem_name, 'Classifier')


def test_delete_problem(setup_db):
    problem_name = 'iris'
    # setup the ramp-kit and ramp-data for the iris challenge
    _setup_ramp_kits_ramp_data(problem_name)
    add_problem(problem_name)
    _check_problem(problem_name, 'Classifier')
    delete_problem(problem_name)
    problems = db.session.query(Problem).all()
    assert len(problems) == 0


def _check_event(event_name, event_title, event_is_public, scores_name):
    events = db.session.query(Event).all()
    assert len(events) == 1
    event = events[0]
    assert event.name == event_name
    assert event.title == event_title
    assert event.is_public == event_is_public

    score_type = db.session.query(EventScoreType, Event).filter(
        Event.name == event_name).all()
    for score, _ in score_type:
        assert score.name in scores_name

    # rebuild the fold indices to check if we stored the right one in the
    # database
    cv_folds = db.session.query(CVFold, Event).filter(
        Event.name == event_name).all()
    X_train, y_train = event.problem.get_train_data()
    cv = event.problem.module.get_cv(X_train, y_train)
    for ((train_indices, test_indices), stored_fold) in zip(cv, cv_folds):
        fold = stored_fold[0]
        assert_array_equal(fold.train_is, train_indices)
        assert_array_equal(fold.test_is, test_indices)


def _setup_event():
    # an event requires a problem
    problem_name = 'iris'
    _setup_problem(problem_name)
    event_name = '{}_test'.format(problem_name)
    event_title = 'test event'
    scores_iris = ('acc', 'error', 'nll', 'f1_70')
    add_event(problem_name, event_name, event_title, is_public=True)
    return event_name


@pytest.mark.parametrize("is_public", [True, False])
def test_add_event(is_public, setup_db):
    problem_name = 'iris'
    _setup_problem(problem_name)

    event_name = '{}_test'.format(problem_name)
    event_title = 'test event'
    scores_iris = ('acc', 'error', 'nll', 'f1_70')
    add_event(problem_name, event_name, event_title, is_public=is_public)
    _check_event(event_name, event_title, is_public, scores_iris)

    err_msg = 'Attempting to overwrite existing event.'
    with pytest.raises(ValueError, match=err_msg):
        add_event(problem_name, event_name, event_title, is_public=is_public,
                  force=False)

    add_event(problem_name, event_name, event_title, is_public=is_public,
              force=True)
    _check_event(event_name, event_title, is_public, scores_iris)


def test_delete_event(setup_db):
    event_name = _setup_event()
    delete_event(event_name)
    event = db.session.query(Event.name==event_name).all()
    assert len(event) == 0


def _setup_sign_up():
    # asking to sign up required a user, a problem, and an event.
    event_name = _setup_event()  # setup the problem and event
    username = 'test_user'
    create_user(
        name=username, password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='asked')
    approve_user(username)
    return event_name, username


def test_ask_sign_up_team(setup_db):
    event_name, username = _setup_sign_up()

    ask_sign_up_team(event_name, username)
    event_team = db.session.query(EventTeam).all()
    assert len(event_team) == 1
    event_team = event_team[0]
    assert event_team.event.name == event_name
    assert event_team.team.name == username
    assert event_team.is_active is True
    assert event_team.last_submission_name is None
    current_datetime = datetime.datetime.now()
    assert event_team.signup_timestamp.year == current_datetime.year
    assert event_team.signup_timestamp.month == current_datetime.month
    assert event_team.signup_timestamp.day == current_datetime.day
    assert event_team.approved is False


def test_sign_up_team(setup_db):
    # asking to sign up required a user, a problem, and an event.
    event_name, username = _setup_sign_up()

    sign_up_team(event_name, username)
    event_team = db.session.query(EventTeam).all()
    assert len(event_team) == 1
    event_team = event_team[0]

    # when signing up a team, the team is approved and the sandbox is setup:
    # the starting kit is submitted without training it.
    assert event_team.last_submission_name == 'starting_kit'
    assert event_team.approved is True
    # check the status of the sandbox submission
    submission = db.session.query(Submission).all()
    assert len(submission) == 1
    submission = submission[0]
    assert submission.name == 'starting_kit'
    assert submission.event_team == event_team
    submission_file = submission.files[0]
    assert submission_file.name == 'classifier'
    assert submission_file.extension == 'py'
    assert (os.path.join('submission_000000001',
                         'classifier.py') in submission_file.path)
    # check the submission on cv fold
    cv_folds = db.session.query(SubmissionOnCVFold).all()
    for fold in cv_folds:
        assert fold.state == 'new'
        assert fold.best is False
        assert fold.contributivity == pytest.approx(0)


def test_get_team_members(setup_db):
    # check the members in a team
    event_name, username = _setup_sign_up()
    _, team, _ = ask_sign_up_team(event_name, username)

    member = list(get_team_members(team))[0]
    assert isinstance(member, User)
    assert member.name == 'test_user'


def test_make_submission_create_new_submission(setup_db):
    # check that we can make a new submission to the database
    # it will require to have already a team and an event
    event_name, username = _setup_sign_up()
    sign_up_team(event_name, username)

    submission_name = 'random_forest_10_10'
    path_submission = os.path.join(
        ramp_config['ramp_kits_path'], 'iris',
        ramp_config['submissions_dir'], submission_name
    )
    make_submission(event_name, username, submission_name, path_submission)
    all_submissions = db.session.query(Submission).all()

    # `sign_up_team` make a submission (sandbox). This submission will be the
    # second submission.
    assert len(all_submissions) == 2
    submission = db.session.query(Submission).filter(
        Submission.name==submission_name).one_or_none()
    assert submission.name == submission_name
    submission_file = submission.files[0]
    assert submission_file.name == 'classifier'
    assert submission_file.extension == 'py'
    assert (os.path.join('submission_000000002',
                         'classifier.py') in submission_file.path)


def test_make_submission_too_early_submission(setup_db):
    # check that we raise an error when the elapsed time was not large enough
    # between the new submission and the previous submission
    event_name, username = _setup_sign_up()
    sign_up_team(event_name, username)

    # check that we have an awaiting time for the event
    event = db.session.query(Event).filter(
        Event.name=='iris_test').one_or_none()
    assert event.min_duration_between_submissions == 900

    # make 2 submissions which are too close from each other
    for submission_idx, submission_name in enumerate(['random_forest_10_10',
                                                      'too_early_submission']):
        path_submission = os.path.join(
            ramp_config['ramp_kits_path'], 'iris',
            ramp_config['submissions_dir'], submission_name
        )
        if submission_idx == 1:
            err_msg = 'You need to wait'
            with pytest.raises(TooEarlySubmissionError, match=err_msg):
                make_submission(event_name, username, submission_name,
                                path_submission)
        else:
            make_submission(event_name, username, submission_name,
                            path_submission)


def test_make_submission_resubmission(setup_db):
    # check that resubmitting the a submission with the same name will raise
    # an error
    event_name, username = _setup_sign_up()
    sign_up_team(event_name, username)

    # submitting the starting_kit which is used as the default submission for
    # the sandbox should raise an error
    submission_name = 'starting_kit'
    path_submission = os.path.join(
            ramp_config['ramp_kits_path'], 'iris',
            ramp_config['submissions_dir'], submission_name
        )

    err_msg = ('Submission "starting_kit" of team "test_user" at event '
               '"iris_test" exists already')
    with pytest.raises(DuplicateSubmissionError, match=err_msg):
        make_submission(event_name, username, submission_name, path_submission)

    # submitting twice a normal submission should raise an error as well
    submission_name = 'random_forest_10_10'
    path_submission = os.path.join(
            ramp_config['ramp_kits_path'], 'iris',
            ramp_config['submissions_dir'], submission_name
        )
    # first submission
    make_submission(event_name, username, submission_name, path_submission)
    # mock that we scored the submission
    submission = (db.session.query(Submission)
                            .filter(Submission.name==submission_name)
                            .one_or_none())
    submission.set_state('scored')
    # # second submission
    err_msg = ('Submission "random_forest_10_10" of team "test_user" at event '
               '"iris_test" exists already')
    with pytest.raises(DuplicateSubmissionError, match=err_msg):
        make_submission(event_name, username, submission_name, path_submission)

    # a resubmission can take place if it is tagged as "new" or failed

    # mock that the submission failed during the training
    submission.set_state('training_error')
    make_submission(event_name, username, submission_name, path_submission)
    # mock that the submissions are new submissions
    submissions = (db.session.query(Submission)
                             .filter(Submission.name==submission_name)
                             .all())
    for submission in submissions:
        submission.set_state('new')
    make_submission(event_name, username, submission_name, path_submission)


def test_make_submission_wrong_submission_files(setup_db):
    # check that we raise an error if the file required by the workflow is not
    # present in the submission or that it has the wrong extension
    event_name, username = _setup_sign_up()
    sign_up_team(event_name, username)

    submission_name  = 'corrupted_submission'
    path_submission = os.path.join(
            ramp_config['ramp_kits_path'], 'iris',
            ramp_config['submissions_dir'], submission_name
        )
    os.makedirs(path_submission)

    # case that there is not files in the submission
    err_msg = 'No file corresponding to the workflow element'
    with pytest.raises(MissingSubmissionFileError, match=err_msg):
        make_submission(event_name, username, submission_name, path_submission)

    # case that there is not file corresponding to the workflow component
    filename = os.path.join(path_submission, 'unknown_file.xxx')
    open(filename, "w+").close()
    err_msg = 'No file corresponding to the workflow element'
    with pytest.raises(MissingSubmissionFileError, match=err_msg):
        make_submission(event_name, username, submission_name, path_submission)

    # case that we have the correct filename but not the right extension
    filename = os.path.join(path_submission, 'classifier.xxx')
    open(filename, "w+").close()
    err_msg = 'All extensions "xxx" are unknown for the submission'
    with pytest.raises(MissingExtensionError, match=err_msg):
        make_submission(event_name, username, submission_name, path_submission)


# TODO: this function should be renamed according if the function db_tools.py
# is renamed
def test_make_submission_and_copy_files(setup_db):
    # check that we can make a submission and that the submission files are
    # copied in the submission folder
    event_name, username = _setup_sign_up()
    sign_up_team(event_name, username)

    submission_name = 'random_forest_10_10'
    path_submission = os.path.join(
        ramp_config['ramp_kits_path'], 'iris',
        ramp_config['submissions_dir'], submission_name
    )
    make_submission_and_copy_files(event_name, username, submission_name,
                                   path_submission)
    submission = (db.session.query(Submission)
                            .filter(Submission.name==submission_name)
                            .one_or_none())
    assert 'submission_000000002' in submission.path
    assert os.path.exists(os.path.join(submission.path, 'classifier.py'))


def test_submit_starting_kit(setup_db):
    event_name, username = _setup_sign_up()
    sign_up_team(event_name, username)
    submit_starting_kit(event_name, username)

    submissions = db.session.query(Submission).all()
    assert len(submissions) == 4
    expected_submission_name = {'starting_kit', 'starting_kit_test',
                                'random_forest_10_10', 'error'}
    submission_name = set(sub.name for sub in submissions)
    assert submission_name == expected_submission_name


@pytest.mark.parametrize(
    "event_name, team_name, user_name, submission_name, err_msg",
    [(None, 'xxx', 'xxx', 'xxx', '"event_name" needs to be specified'),
     ('iris_test', 'test_user', 'test_user', None, 'If "user_name" is given'),
     ('iris_test', None, None, 'starting_kit_test', '"submission_name" is')]
)
def test_get_submissions_error(event_name, team_name, user_name,
                               submission_name, err_msg, setup_toy_db):
    with pytest.raises(ValueError, match=err_msg):
        get_submissions(event_name=event_name, team_name=team_name,
                        user_name=user_name, submission_name=submission_name)


@pytest.mark.parametrize(
    "event_name, team_name, user_name, submission_name, expected_n_sub",
    [(None, None, None, None, 14),
     ('iris_test', None, None, None, 8),
     ('boston_housing_test', None, None, None, 6),
     ('iris_test', 'test_user', None, None, 4),
     ('boston_housing_test', 'test_user', None, None, 3),
     ('iris_test', None, 'test_user', None, 4),
     ('iris_test', 'test_user', None, 'starting_kit_test', 1)]
)
def test_get_submissions(event_name, team_name, user_name, submission_name,
                         expected_n_sub, setup_toy_db):
    submissions = get_submissions(event_name=event_name,
                                  team_name=team_name,
                                  user_name=user_name,
                                  submission_name=submission_name)
    assert len(submissions) == expected_n_sub


def test_get_new_submissions(setup_toy_db):
    submissions = get_new_submissions('iris_test')
    assert len(submissions) == 8
    # mark half of the submissions as trained
    for submission_idx in range(4):
        submissions[submission_idx].state = 'trained'
    submissions = get_new_submissions('iris_test')
    assert len(submissions) == 4
    # mark the remaining submissions as trained
    for submission in submissions:
        submission.state = 'trained'
    submissions = get_new_submissions('iris_test')
    assert len(submissions) == 0


def test_submissions_on_cv_fold(setup_toy_db):
    submission = get_submissions(event_name='iris_test',
                                 team_name='test_user',
                                 submission_name='starting_kit_test')[0]
    submissions_cv = get_submission_on_cv_folds(submission.id)
    assert len(submissions_cv) == 2
    for sub_cv in submissions_cv:
        assert sub_cv.submission.name == submission.name
        assert sub_cv.submission_id == submission.id
    cv_fold_id = [sub_cv.cv_fold_id for sub_cv in submissions_cv]
    assert all(cv_fold_id[i-1] < cv_fold_id[i]
               for i in range(1, len(cv_fold_id)))


def test_update_submission_on_cv_fold(setup_toy_db):
    submission = get_submissions(event_name='iris_test',
                                 team_name='test_user',
                                 submission_name='starting_kit_test')[0]
    submissions_cv = get_submission_on_cv_folds(submission.id)
    for cv_fold_idx, cv_fold in enumerate(submissions_cv):
        # define the path for this results
        path_results = os.path.join(HERE, 'data', 'iris_predictions',
                                    'fold_{}'.format(cv_fold_idx))
        update_submission_on_cv_fold(cv_fold, path_results)
    submissions_cv = get_submission_on_cv_folds(submission.id)
    for cv_fold_idx, cv_fold in enumerate(submissions_cv):
        path_results = os.path.join(HERE, 'data', 'iris_predictions',
                                    'fold_{}'.format(cv_fold_idx))
        # check only the training time
        train_time = np.asscalar(
            np.loadtxt(os.path.join(path_results, 'train_time'))
        )
        assert cv_fold.train_time == pytest.approx(train_time)
        # check only the training predictions
        y_pred_train = np.load(os.path.join(path_results, 'y_pred_train.npz'))
        assert_allclose(cv_fold.full_train_y_pred, y_pred_train['y_pred'])
        # check only the accuracy on the training
        scores = pd.read_csv(os.path.join(path_results, 'scores.csv'),
                             index_col=0)
        for score in cv_fold.scores:
            score_type = score.name
            assert score.train_score == pytest.approx(scores.loc['train',
                                                                 score_type])
            assert score.valid_score == pytest.approx(scores.loc['valid',
                                                                 score_type])
            assert score.test_score == pytest.approx(scores.loc['test',
                                                                score_type])
