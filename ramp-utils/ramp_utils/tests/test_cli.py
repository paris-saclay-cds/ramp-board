from difflib import SequenceMatcher
import shutil

from click.testing import CliRunner

from ramp_database.utils import setup_db
from ramp_database.model import Model

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_utils.cli import main


def teardown_function(function):
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    # FIXME: we are recreating the deployment directory but it should be
    # replaced by an temporary creation of folder.
    match = SequenceMatcher(
        None,
        ramp_config['ramp']['kit_dir'],
        ramp_config['ramp']['submissions_dir']
    ).find_longest_match(
        0, len(ramp_config['ramp']['kit_dir']),
        0, len(ramp_config['ramp']['submissions_dir'])
    )
    deployment_dir = ramp_config['ramp']['kit_dir'][match.a:
                                                    match.a + match.size]
    shutil.rmtree(deployment_dir, ignore_errors=True)
    db, _ = setup_db(database_config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_deploy_ramp_event():
    runner = CliRunner()
    result = runner.invoke(main, ['deploy-ramp-event',
                                  '--config', database_config_template(),
                                  '--event-config', ramp_config_template()])
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ['deploy-ramp-event',
                                  '--config', database_config_template(),
                                  '--event-config', ramp_config_template(),
                                  '--force'])
    assert result.exit_code == 0, result.output
