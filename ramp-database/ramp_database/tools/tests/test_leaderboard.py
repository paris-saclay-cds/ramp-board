import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_engine.dispatcher import Dispatcher

from ramp_database.model import Model

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.model import EventTeam

from ramp_database.tools.event import get_event
from ramp_database.tools.team import get_event_team_by_name

from ramp_database.tools.leaderboard import get_leaderboard
from ramp_database.tools.leaderboard import update_all_user_leaderboards
from ramp_database.tools.leaderboard import update_leaderboards
from ramp_database.tools.leaderboard import update_user_leaderboards


@pytest.fixture(scope='module')
def session_toy_db(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


@pytest.fixture
def session_toy_function(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_update_leaderboard_functions(session_toy_function):
    event_name = 'iris_test'
    user_name = 'test_user'
    for leaderboard_type in ['public', 'private', 'failed',
                             'public competition', 'private competition']:
        leaderboard = get_leaderboard(session_toy_function, leaderboard_type,
                                      event_name)
        assert leaderboard is None
    leaderboard = get_leaderboard(session_toy_function, 'new', event_name)
    assert leaderboard

    event = get_event(session_toy_function, event_name)
    assert event.private_leaderboard_html is None
    assert event.public_leaderboard_html_with_links is None
    assert event.public_leaderboard_html_no_links is None
    assert event.failed_leaderboard_html is None
    assert event.public_competition_leaderboard_html is None
    assert event.private_competition_leaderboard_html is None
    assert event.new_leaderboard_html

    event_team = get_event_team_by_name(session_toy_function, event_name,
                                        user_name)
    assert event_team.leaderboard_html is None
    assert event_team.failed_leaderboard_html is None
    assert event_team.new_leaderboard_html

    event_teams = (session_toy_function.query(EventTeam)
                                       .filter_by(event=event)
                                       .all())
    for et in event_teams:
        assert et.leaderboard_html is None
        assert et.failed_leaderboard_html is None
        assert et.new_leaderboard_html

    # run the dispatcher to process the different submissions
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(
        config, event_config, n_workers=-1, hunger_policy='exit'
    )
    dispatcher.launch()
    session_toy_function.commit()

    update_leaderboards(session_toy_function, event_name)
    event = get_event(session_toy_function, event_name)
    assert event.private_leaderboard_html
    assert event.public_leaderboard_html_with_links
    assert event.public_leaderboard_html_no_links
    assert event.failed_leaderboard_html
    assert event.public_competition_leaderboard_html
    assert event.private_competition_leaderboard_html
    assert event.new_leaderboard_html is None

    update_user_leaderboards(session_toy_function, event_name, user_name)
    event_team = get_event_team_by_name(session_toy_function, event_name,
                                        user_name)
    assert event_team.leaderboard_html
    assert event_team.failed_leaderboard_html
    assert event_team.new_leaderboard_html is None

    update_all_user_leaderboards(session_toy_function, event_name)
    event_teams = (session_toy_function.query(EventTeam)
                                       .filter_by(event=event)
                                       .all())
    for et in event_teams:
        assert et.leaderboard_html
        assert et.failed_leaderboard_html
        assert et.new_leaderboard_html is None


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


def test_get_leaderboard(session_toy_db):
    leaderboard_new = get_leaderboard(session_toy_db, 'new', 'iris_test')
    assert leaderboard_new.count('<tr>') == 6
    leaderboard_new = get_leaderboard(session_toy_db, 'new', 'iris_test',
                                      'test_user')
    assert leaderboard_new.count('<tr>') == 3

    # run the dispatcher to process the different submissions
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(
        config, event_config, n_workers=-1, hunger_policy='exit'
    )
    dispatcher.launch()
    session_toy_db.commit()

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
    assert """<th>submission ID</th>
      <th>team</th>
      <th>submission</th>
      <th>bag public acc</th>
      <th>mean public acc</th>
      <th>std public acc</th>
      <th>bag public error</th>
      <th>mean public error</th>
      <th>std public error</th>
      <th>bag public nll</th>
      <th>mean public nll</th>
      <th>std public nll</th>
      <th>bag public f1_70</th>
      <th>mean public f1_70</th>
      <th>std public f1_70</th>
      <th>bag private acc</th>
      <th>mean private acc</th>
      <th>std private acc</th>
      <th>bag private error</th>
      <th>mean private error</th>
      <th>std private error</th>
      <th>bag private nll</th>
      <th>mean private nll</th>
      <th>std private nll</th>
      <th>bag private f1_70</th>
      <th>mean private f1_70</th>
      <th>std private f1_70</th>
      <th>train time [s]</th>
      <th>validation time [s]</th>
      <th>test time [s]</th>
      <th>max RAM [MB]</th>
      <th>submitted at (UTC)</th>""" in leaderboard_private
    assert """<th>team</th>
      <th>submission</th>
      <th>acc</th>
      <th>error</th>
      <th>nll</th>
      <th>f1_70</th>
      <th>train time [s]</th>
      <th>validation time [s]</th>
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
      <th>validation time [s]</th>
      <th>submitted at (UTC)</th>""" in competition_public
    assert """<th>rank</th>
      <th>move</th>
      <th>team</th>
      <th>submission</th>
      <th>acc</th>
      <th>train time [s]</th>
      <th>validation time [s]</th>
      <th>test time [s]</th>
      <th>submitted at (UTC)</th>""" in competition_private
