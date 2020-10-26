import os
import shutil
from time import sleep
from contextlib import contextmanager

import pytest

from ramp_engine.local import CondaEnvWorker
from ramp_engine.remote import RemoteWorker
from ramp_engine.conda import _conda_info_envs

ALL_WORKERS = [CondaEnvWorker, RemoteWorker]


def _is_conda_env_installed():
    # we required a "ramp-iris" conda environment to run the test. Check if it
    # is available. Otherwise skip all tests.
    try:
        conda_info = _conda_info_envs()
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


@contextmanager
def get_conda_worker(submission_name, Worker=CondaEnvWorker,
                     conda_env='ramp-iris'):

    module_path = os.path.dirname(__file__)
    config = {'kit_dir': os.path.join(module_path, 'kits', 'iris'),
              'data_dir': os.path.join(module_path, 'kits', 'iris'),
              'submissions_dir': os.path.join(module_path, 'kits',
                                              'iris', 'submissions'),
              'logs_dir': os.path.join(module_path, 'kits', 'iris', 'log'),
              'predictions_dir': os.path.join(
                  module_path, 'kits', 'iris', 'predictions'),
              'conda_env': conda_env}

    if issubclass(Worker, RemoteWorker):
        pytest.importorskip('dask')
        pytest.importorskip('dask.distributed')
        config['dask_scheduler'] = None

    worker = Worker(config=config, submission='starting_kit')
    yield worker
    # remove all directories that we potentially created
    _remove_directory(worker)

    if issubclass(Worker, RemoteWorker) and hasattr(worker, '_client'):
        worker._client.close()


def _remove_directory(worker):
    if 'kit_dir' not in worker.config:
        return
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
@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_launch(submission, Worker):
    with get_conda_worker(submission, Worker=Worker) as worker:
        worker.launch()
        # check that teardown removed the predictions
        output_training_dir = os.path.join(worker.config['kit_dir'],
                                           'submissions',
                                           worker.submission,
                                           'training_output')
        assert not os.path.exists(output_training_dir), \
            "teardown() failed to remove the predictions"


@pytest.mark.parametrize("submission", ('starting_kit', 'random_forest_10_10'))
@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker(submission, Worker):
    with get_conda_worker(submission, Worker=Worker) as worker:
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


@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_without_conda_env_specified(Worker):
    with get_conda_worker('starting_kit', Worker=Worker) as worker:
        # remove the conda_env parameter from the configuration
        del worker.config['conda_env']
        # if the conda environment is not given in the configuration, we should
        # fall back on the base environment of conda
        # the conda environment is set during setup; thus no need to launch
        # submission
        worker.setup()
        assert 'envs' not in worker._python_bin_path


@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_error_missing_config_param(Worker):
    with get_conda_worker('starting_kit', Worker=Worker) as worker:
        # we remove one of the required parameter
        del worker.config['kit_dir']

        err_msg = "The worker required the parameter 'kit_dir'"
        with pytest.raises(ValueError, match=err_msg):
            worker.setup()


@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_error_unknown_env(Worker):
    with get_conda_worker(
        'starting_kit', conda_env='xxx', Worker=Worker
    ) as worker:
        msg_err = "The specified conda environment xxx does not exist."
        with pytest.raises(ValueError, match=msg_err):
            worker.setup()
            assert worker.status == 'error'


@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_error_multiple_launching(Worker):
    submission = 'starting_kit'
    with get_conda_worker(submission, Worker=Worker) as worker:

        worker.setup()
        worker.launch_submission()
        err_msg = "Wait that the submission is processed"
        with pytest.raises(ValueError, match=err_msg):
            worker.launch_submission()
        # force to wait for the submission to be processed
        worker.collect_results()


@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_error_soon_teardown(Worker):
    with get_conda_worker('starting_kit', Worker=Worker) as worker:
        worker.setup()
        err_msg = 'Collect the results before to kill the worker.'
        with pytest.raises(ValueError, match=err_msg):
            worker.teardown()


@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_error_soon_collection(Worker):
    with get_conda_worker('starting_kit', Worker=Worker) as worker:
        err_msg = r"Call the method setup\(\) and launch_submission\(\) before"
        with pytest.raises(ValueError, match=err_msg):
            worker.collect_results()
        worker.setup()
        err_msg = r"Call the method launch_submission\(\)"
        with pytest.raises(ValueError, match=err_msg):
            worker.collect_results()


@pytest.mark.parametrize("Worker", ALL_WORKERS)
def test_conda_worker_timeout(Worker):
    with get_conda_worker('random_forest_10_10', Worker=Worker) as worker:
        worker.config['timeout'] = 1

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
