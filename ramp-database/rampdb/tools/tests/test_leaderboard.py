import datetime
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampbkd.dispatcher import Dispatcher

from rampdb.model import Model

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.leaderboard import get_leaderboard


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_toy_db(config):
    try:
        create_toy_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, _ = setup_db(config['sqlalchemy'])
        Model.metadata.drop_all(db)


@pytest.mark.parametrize(
    'leaderboard_type, expected_html',
    [('new', not None),
     ('public', None),
     ('private', None),
     ('failed', None),
     ('public competition', None),
     ('private competition', None)]
)
def test_get_leaderboard_only_new_submissions(session_toy_db, leaderboard_type,
                                              expected_html):
    # only check that the submission should be shown as new when the
    # dispatcher was not started.
    if expected_html is not None:
        assert get_leaderboard(session_toy_db, leaderboard_type, 'iris_test')
    else:
        assert (get_leaderboard(
            session_toy_db, leaderboard_type, 'iris_test') is expected_html)


def test_get_leaderboard(session_toy_db, config):
    leaderboard_new = get_leaderboard(session_toy_db, 'new', 'iris_test')
    assert leaderboard_new.count('<tr>') == 6
    leaderboard_new = get_leaderboard(session_toy_db, 'new', 'iris_test',
                                      'test_user')
    assert leaderboard_new.count('<tr>') == 3

    # run the dispatcher to process the different submissions
    dispatcher = Dispatcher(config, n_worker=-1, hunger_policy='exit')
    dispatcher.launch()

    assert get_leaderboard(session_toy_db, 'new', 'iris_test') is None
    # the iris dataset has a single submission which is failing
    leaderboard_failed = get_leaderboard(session_toy_db, 'failed', 'iris_test')
    assert leaderboard_failed.count('<tr>') == 2
    leaderboard_failed = get_leaderboard(session_toy_db, 'failed', 'iris_test',
                                         'test_user')
    assert leaderboard_failed.count('<tr>') == 1

    # the remaining submission should be successful
    leaderboard_public = get_leaderboard(session_toy_db, 'public', 'iris_test')
    assert leaderboard_public.count('<tr>') == 4
    leaderboard_public = get_leaderboard(session_toy_db, 'public', 'iris_test',
                                         'test_user')
    assert leaderboard_public.count('<tr>') == 2

    leaderboard_private = get_leaderboard(session_toy_db, 'private',
                                          'iris_test')
    assert leaderboard_private.count('<tr>') == 4
    leaderboard_private = get_leaderboard(session_toy_db, 'private',
                                          'iris_test', 'test_user')
    assert leaderboard_private.count('<tr>') == 2

    # the competition leaderboard will have the best solution for each user
    competition_public = get_leaderboard(session_toy_db, 'public competition',
                                         'iris_test')
    assert competition_public.count('<tr>') == 2
    competition_private = get_leaderboard(session_toy_db,
                                          'private competition', 'iris_test')
    assert competition_private.count('<tr>') == 2

    # check the difference between the public and private leaderboard
    assert leaderboard_private.count('<td>') > leaderboard_public.count('<td>')
    for private_term in ['bag', 'mean', 'std', 'private']:
        assert private_term not in leaderboard_public
        assert private_term in leaderboard_private

    # check the column name in each leaderboard
    assert """<th>team</th>
      <th>submission</th>
      <th>bag public acc</th>
      <th>bag public error</th>
      <th>bag public nll</th>
      <th>bag public f1_70</th>
      <th>bag private acc</th>
      <th>bag private error</th>
      <th>bag private nll</th>
      <th>bag private f1_70</th>
      <th>mean public acc</th>
      <th>mean public error</th>
      <th>mean public nll</th>
      <th>mean public f1_70</th>
      <th>mean private acc</th>
      <th>mean private error</th>
      <th>mean private nll</th>
      <th>mean private f1_70</th>
      <th>std public acc</th>
      <th>std public error</th>
      <th>std public nll</th>
      <th>std public f1_70</th>
      <th>std private acc</th>
      <th>std private error</th>
      <th>std private nll</th>
      <th>std private f1_70</th>
      <th>contributivity</th>
      <th>historical contributivity</th>
      <th>train time [s]</th>
      <th>test time [s]</th>
      <th>max RAM [MB]</th>
      <th>submitted at (UTC)</th>""" in leaderboard_private
    assert """<th>team</th>
      <th>submission</th>
      <th>acc</th>
      <th>error</th>
      <th>nll</th>
      <th>f1_70</th>
      <th>contributivity</th>
      <th>historical contributivity</th>
      <th>train time [s]</th>
      <th>test time [s]</th>
      <th>max RAM [MB]</th>
      <th>submitted at (UTC)</th>""" in leaderboard_public
    assert """<th>team</th>
      <th>submission</th>
      <th>submitted at (UTC)</th>
      <th>error</th>""" in leaderboard_failed

    # check the same for the competition leaderboard
    assert """<th>rank</th>
      <th>team</th>
      <th>submission</th>
      <th>acc</th>
      <th>train time [s]</th>
      <th>test time [s]</th>
      <th>submitted at (UTC)</th>""" in competition_public
    assert """<th>rank</th>
      <th>move</th>
      <th>team</th>
      <th>submission</th>
      <th>acc</th>
      <th>train time [s]</th>
      <th>test time [s]</th>
      <th>submitted at (UTC)</th>""" in competition_private
