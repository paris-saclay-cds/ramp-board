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
def session_toy_db():
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
def session_toy_function():
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
    for private_term in ['bagged', 'mean', 'std', 'test time']:
        assert private_term not in leaderboard_public
        assert private_term in leaderboard_private

    # check the column name in each leaderboard
    assert """<th>team</th>
      <th>submission</th>
      <th>bagged test acc</th>
      <th>mean test acc</th>
      <th>std test acc</th>
      <th>bagged valid acc</th>
      <th>mean valid acc</th>
      <th>std valid acc</th>
      <th>bagged test error</th>
      <th>mean test error</th>
      <th>std test error</th>
      <th>bagged valid error</th>
      <th>mean valid error</th>
      <th>std valid error</th>
      <th>bagged test nll</th>
      <th>mean test nll</th>
      <th>std test nll</th>
      <th>bagged valid nll</th>
      <th>mean valid nll</th>
      <th>std valid nll</th>
      <th>bagged test f1_70</th>
      <th>mean test f1_70</th>
      <th>std test f1_70</th>
      <th>bagged valid f1_70</th>
      <th>mean valid f1_70</th>
      <th>std valid f1_70</th>
      <th>contributivity</th>
      <th>historical contributivity</th>
      <th>train time [s]</th>
      <th>valid time [s]</th>
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
      <th>valid time [s]</th>
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
      <th>valid time [s]</th>
      <th>submitted at (UTC)</th>""" in competition_public
    assert """<th>rank</th>
      <th>move</th>
      <th>team</th>
      <th>submission</th>
      <th>acc</th>
      <th>train time [s]</th>
      <th>valid time [s]</th>
      <th>submitted at (UTC)</th>""" in competition_private
