import shutil
import subprocess

import pytest

from rampwf.workflows import FeatureExtractorClassifier

from databoard import db
from databoard import deployment_path

from databoard.model import Problem
from databoard.model import User
from databoard.model import Workflow
from databoard.model import WorkflowElement
from databoard.model import WorkflowElementType
from databoard.model import NameClashError

from databoard.testing import create_test_db
from databoard.testing import _setup_ramp_kits_ramp_data
from databoard.utils import check_password

from databoard.db_tools import create_user
from databoard.db_tools import approve_user
from databoard.db_tools import add_workflow
from databoard.db_tools import add_problem
from databoard.db_tools import delete_problem


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
    # TODO: add a team with the name of a user to trigger an error


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


def test_add_problem(setup_db):
    # setup the ramp-kit and ramp-data for the iris challenge
    _setup_ramp_kits_ramp_data('iris')
    add_problem('iris')
    problems = db.session.query(Problem).all()
    assert len(problems) == 1
    problem = problems[0]
    assert problem.workflow.name == 'Classifier'


def test_delete_problem(setup_db):
    # setup the ramp-kit and ramp-data for the iris challenge
    _setup_ramp_kits_ramp_data('iris')
    add_problem('iris')
    delete_problem('iris')
