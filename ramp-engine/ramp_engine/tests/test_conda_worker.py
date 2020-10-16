import json
import os
import subprocess
import shutil
from time import sleep

import pytest

from ramp_engine.local import CondaEnvWorker


def _is_conda_env_installed():
    # we required a "ramp-iris" conda environment to run the test. Check if it
    # is available. Otherwise skip all tests.
    try:
        proc = subprocess.Popen(
            ["conda", "info", "--envs", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, _ = proc.communicate()
        conda_info = json.loads(stdout)
        envs_path = conda_info['envs'][1:]
        if not envs_path or not any(['ramp-iris' in env for env in envs_path]):
            return True
        return False
    except:  # noqa
        # conda is not installed
        return True


pytestmark = pytest.mark.skipif(
    _is_conda_env_installed(),
    reason=('CondaEnvWorker required conda and an environment named '
            '"ramp-iris". No such environment was found. Check the '
            '"ci_tools/environment_iris_kit.yml" file to create such '
            'environment.')
)


@pytest.fixture
def get_conda_worker():
    def _create_worker(submission_name, conda_env='ramp-iris'):
        module_path = os.path.dirname(__file__)
        config = {'kit_dir': os.path.join(module_path, 'kits', 'iris'),
                  'data_dir': os.path.join(module_path, 'kits', 'iris'),
                  'submissions_dir': os.path.join(module_path, 'kits',
                                                  'iris', 'submissions'),
                  'logs_dir': os.path.join(module_path, 'kits', 'iris', 'log'),
                  'predictions_dir': os.path.join(
                      module_path, 'kits', 'iris', 'predictions'),
                  'conda_env': conda_env}
        return CondaEnvWorker(config=config, submission='starting_kit')
    return _create_worker


def _remove_directory(worker):
    output_training_dir = os.path.join(
        worker.config['kit_dir'], 'submissions', worker.submission,
        'training_output'
    )
    for directory in (output_training_dir,
                      worker.config['logs_dir'],
                      worker.config['predictions_dir']):
        if os.path.exists(directory):
            shutil.rmtree(directory)


@pytest.mark.parametrize("submission", ('starting_kit', 'random_forest_10_10'))
def test_conda_worker_launch(submission, get_conda_worker):
    worker = get_conda_worker(submission)
    try:
        worker.launch()
        # check that teardown removed the predictions
        output_training_dir = os.path.join(worker.config['kit_dir'],
                                           'submissions',
                                           worker.submission,
                                           'training_output')
        assert not os.path.exists(output_training_dir), \
            "teardown() failed to remove the predictions"
    finally:
        # remove all directories that we potentially created
        _remove_directory(worker)


@pytest.mark.parametrize("submission", ('starting_kit', 'random_forest_10_10'))
def test_conda_worker(submission, get_conda_worker):
    worker = get_conda_worker(submission)
    try:
        assert worker.status == 'initialized'
        worker.setup()
        assert worker.status == 'setup'
        worker.launch_submission()
        assert worker.status == 'running'
        exit_status, _ = worker.collect_results()
        assert exit_status == 0
        assert worker.status == 'collected'
        worker.teardown()
        # check that teardown removed the predictions
        output_training_dir = os.path.join(worker.config['kit_dir'],
                                           'submissions',
                                           worker.submission,
                                           'training_output')
        assert not os.path.exists(output_training_dir), \
            "teardown() failed to remove the predictions"
    finally:
        # remove all directories that we potentially created
        _remove_directory(worker)


def test_conda_worker_without_conda_env_specified(get_conda_worker):
    worker = get_conda_worker('starting_kit')
    # remove the conda_env parameter from the configuration
    del worker.config['conda_env']
    # if the conda environment is not given in the configuration, we should
    # fall back on the base environment of conda
    # the conda environment is set during setup; thus no need to launch
    # submission
    worker.setup()
    assert 'envs' not in worker._python_bin_path


def test_conda_worker_error_missing_config_param(get_conda_worker):
    worker = get_conda_worker('starting_kit')
    # we remove one of the required parameter
    del worker.config['kit_dir']

    err_msg = "The worker required the parameter 'kit_dir'"
    with pytest.raises(ValueError, match=err_msg):
        worker.setup()


def test_conda_worker_error_unknown_env(get_conda_worker):
    worker = get_conda_worker('starting_kit', conda_env='xxx')
    msg_err = "The specified conda environment xxx does not exist."
    with pytest.raises(ValueError, match=msg_err):
        worker.setup()
        assert worker.status == 'error'


def test_conda_worker_error_multiple_launching(get_conda_worker):
    submission = 'starting_kit'
    worker = get_conda_worker(submission)
    try:
        worker.setup()
        worker.launch_submission()
        err_msg = "Wait that the submission is processed"
        with pytest.raises(ValueError, match=err_msg):
            worker.launch_submission()
        # force to wait for the submission to be processed
        worker.collect_results()
    finally:
        # remove all directories that we potentially created
        _remove_directory(worker)


def test_conda_worker_error_soon_teardown(get_conda_worker):
    worker = get_conda_worker('starting_kit')
    worker.setup()
    err_msg = 'Collect the results before to kill the worker.'
    with pytest.raises(ValueError, match=err_msg):
        worker.teardown()


def test_conda_worker_error_soon_collection(get_conda_worker):
    worker = get_conda_worker('starting_kit')
    err_msg = r"Call the method setup\(\) and launch_submission\(\) before"
    with pytest.raises(ValueError, match=err_msg):
        worker.collect_results()
    worker.setup()
    err_msg = r"Call the method launch_submission\(\)"
    with pytest.raises(ValueError, match=err_msg):
        worker.collect_results()


def test_conda_worker_timeout(get_conda_worker):
    worker = get_conda_worker('random_forest_10_10')
    worker.config['timeout'] = 1
    try:
        assert worker.status == 'initialized'
        worker.setup()
        assert worker.status == 'setup'
        worker.launch_submission()
        assert not worker.check_timeout()
        assert worker.status == 'running'
        sleep(2)
        assert worker.check_timeout() is True
        assert worker.status == 'timeout'
        exit_status, _ = worker.collect_results()
        assert exit_status > 0
        assert worker.status == 'collected'
        worker.teardown()
        # check that teardown removed the predictions
        output_training_dir = os.path.join(worker.config['kit_dir'],
                                           'submissions',
                                           worker.submission,
                                           'training_output')
        assert not os.path.exists(output_training_dir), \
            "teardown() failed to remove the predictions"
    finally:
        # remove all directories that we potentially created
        _remove_directory(worker)
