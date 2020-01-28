import os
import shutil

import pytest
import yaml

from click.testing import CliRunner

from ramp_utils import read_config
from ramp_utils import generate_ramp_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.utils import setup_db
from ramp_database.model import Model
from ramp_database.testing import create_toy_db

from ramp_database.cli import main

from ramp_utils.cli import main as main_utils


@pytest.fixture(scope="module")
def make_toy_db(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        yield
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_add_user(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['add-user',
                                  '--config', database_config_template(),
                                  '--login', 'glemaitre',
                                  '--password', 'xxx',
                                  '--lastname', 'xxx',
                                  '--firstname', 'xxx',
                                  '--email', 'xxx',
                                  '--access-level', 'user'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_delete_user(make_toy_db):
    runner = CliRunner()
    runner.invoke(main, ['add-user',
                         '--config', database_config_template(),
                         '--login', 'yyy',
                         '--password', 'yyy',
                         '--lastname', 'yyy',
                         '--firstname', 'yyy',
                         '--email', 'yyy',
                         '--access-level', 'user'],
                  catch_exceptions=False)
    result = runner.invoke(main, ['delete-user',
                                  '--config', database_config_template(),
                                  '--login', 'yyy'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_approve_user(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['approve-user',
                                  '--config', database_config_template(),
                                  '--login', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_make_user_admin(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['make-user-admin',
                                  '--config', database_config_template(),
                                  '--login', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_set_user_access_level(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['set-user-access-level',
                                  '--config', database_config_template(),
                                  '--login', 'glemaitre',
                                  '--access-level', 'admin'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_add_problem(make_toy_db):
    runner = CliRunner()
    ramp_config = generate_ramp_config(read_config(ramp_config_template()))
    result = runner.invoke(main, ['add-problem',
                                  '--config', database_config_template(),
                                  '--problem', 'iris',
                                  '--kit-dir', ramp_config['ramp_kit_dir'],
                                  '--data-dir', ramp_config['ramp_data_dir'],
                                  '--force', True],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_add_event(make_toy_db):
    runner = CliRunner()
    ramp_config = generate_ramp_config(read_config(ramp_config_template()))
    result = runner.invoke(main, ['add-event',
                                  '--config', database_config_template(),
                                  '--problem', 'iris',
                                  '--event', 'iris_test',
                                  '--title', 'Iris classification',
                                  '--sandbox', ramp_config['sandbox_name'],
                                  '--submissions-dir',
                                  ramp_config['ramp_submissions_dir'],
                                  '--is-public', False,
                                  '--force', True],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "config_event, from_disk, force, expected_msg",
    [("events/iris_test/config2.yml", False,
      False, 'No such file: events/iris_test/config2'),
     ("/tmp/databoard_test/events/iris_test/config.yml", False,
      False, 'please use options --force --from_disk'),
     ("/tmp/databoard_test/events/iris_test/config.yml", True,
      True, '')]
)
def test_delete_event(make_toy_db, config_event, from_disk,
                      force, expected_msg):
    runner = CliRunner()
    ramp_config = read_config(ramp_config_template())

    deployment_dir = os.path.commonpath([ramp_config['ramp']['kit_dir'],
                                        ramp_config['ramp']['data_dir']])

    result = runner.invoke(main_utils, ['init-event',
                                        '--name', 'iris_test',
                                        '--deployment-dir', deployment_dir])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main_utils, ['deploy-event',
                                        '--config',
                                        database_config_template(),
                                        '--event-config',
                                        ramp_config_template(),
                                        '--force'])

    if 'No such file' not in expected_msg:
        with open(config_event, 'w') as file:
            yaml.dump(ramp_config, file)

    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ['delete-event',
                                  '--config', database_config_template(),
                                  '--config-event', config_event,
                                  '--from-disk', from_disk,
                                  '--force', force],
                           catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert expected_msg in result.output

    if force and from_disk:
        assert not os.path.exists(config_event)


def test_sign_up_team(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['sign-up-team',
                                  '--config', database_config_template(),
                                  '--event', 'iris_test',
                                  '--team', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_add_event_admin(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['add-event-admin',
                                  '--config', database_config_template(),
                                  '--event', 'iris_test',
                                  '--user', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_add_submission(make_toy_db):
    ramp_config = generate_ramp_config(read_config(ramp_config_template()))
    submission_name = 'new_submission'
    submission_path = os.path.join(ramp_config['ramp_kit_submissions_dir'],
                                   submission_name)
    shutil.copytree(
        os.path.join(ramp_config['ramp_kit_submissions_dir'],
                     'random_forest_10_10'),
        submission_path
    )
    runner = CliRunner()
    result = runner.invoke(main, ['add-submission',
                                  '--config', database_config_template(),
                                  '--event', 'iris_test',
                                  '--team', 'glemaitre',
                                  '--submission', submission_name,
                                  '--path', submission_path],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_get_submission_by_state(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['get-submissions-by-state',
                                  '--config', database_config_template(),
                                  '--event', 'boston_housing_test',
                                  '--state', 'new'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "ID" in result.output
    result = runner.invoke(main, ['get-submissions-by-state',
                                  '--config', database_config_template(),
                                  '--event', 'iris_test',
                                  '--state', 'scored'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert 'No submission for this event and this state' in result.output


def test_set_submission_state(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['set-submission-state',
                                  '--config', database_config_template(),
                                  '--submission-id', 3,
                                  '--state', 'scored'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_update_leaderboards(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['update-leaderboards',
                                  '--config', database_config_template(),
                                  '--event', 'iris_test'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_update_user_leaderboards(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['update-user-leaderboards',
                                  '--config', database_config_template(),
                                  '--event', 'iris_test',
                                  '--user', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_update_all_user_leaderboards(make_toy_db):
    runner = CliRunner()
    result = runner.invoke(main, ['update-all-users-leaderboards',
                                  '--config', database_config_template(),
                                  '--event', 'iris_test'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output
