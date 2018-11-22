import os
import subprocess
import shutil

import pytest

from rampbkd.local import CondaEnvWorker


@pytest.fixture
def get_conda_worker():
    def _create_worker(submission_name, conda_env='ramp'):
        module_path = os.path.dirname(__file__)
        ramp_kit_dir = os.path.join(module_path, 'kits', 'iris')
        ramp_data_dir = ramp_kit_dir
        config = {'ramp_kit_dir': os.path.join(module_path, 'kits', 'iris'),
                  'ramp_data_dir': os.path.join(module_path, 'kits', 'iris'),
                  'local_log_folder': os.path.join(
                      module_path, 'kits', 'iris', 'log'),
                  'local_predictions_folder': os.path.join(
                      module_path, 'kits', 'iris', 'predictions'),
                  'conda_env': conda_env}
        return CondaEnvWorker(config=config, submission='starting_kit')
    return _create_worker


def _remove_directory(worker):
    output_training_dir = os.path.join(worker.config['ramp_kit_dir'],
                                       'submissions',
                                       worker.submission,
                                       'training_output')
    log_dir = os.path.join(worker.config['local_log_folder'])
    pred_dir = os.path.join(worker.config['local_predictions_folder'],
                            worker.submission)
    for directory in (output_training_dir,
                      worker.config['local_log_folder'],
                      worker.config['local_predictions_folder']):
        if os.path.exists(directory):
            shutil.rmtree(directory)


@pytest.mark.parametrize("submission", ('starting_kit', 'random_forest_10_10'))
def test_conda_worker(submission, get_conda_worker):
    worker = get_conda_worker(submission)
    try:
        assert worker.status == 'initialized'
        worker.setup()
        assert worker.status == 'setup'
        worker.launch_submission()
        assert worker.status == 'running'
        print(worker.collect_results())
        assert worker.status == 'collected'
        worker.teardown()
        # check that teardown removed the predictions
        output_training_dir = os.path.join(worker.config['ramp_kit_dir'],
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
    # remove the conva_env parameter from the configuration
    del worker.config['conda_env']
    # the conda environment is set during setup; thus no need to launch
    # submission
    worker.setup()


def test_conda_worker_error_missing_config_param(get_conda_worker):
    worker = get_conda_worker('starting_kit')
    # we remove one of the required parameter
    del worker.config['ramp_kit_dir']

    err_msg = "The worker required the parameter 'ramp_kit_dir'"
    with pytest.raises(ValueError, match=err_msg):
        worker.setup()


def test_conda_worker_error_unknown_env(get_conda_worker):
    worker = get_conda_worker('starting_kit', conda_env='xxx')
    msg_err = "The specified conda environment xxx does not exist."
    with pytest.raises(ValueError, match=msg_err):
        worker.setup()


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
