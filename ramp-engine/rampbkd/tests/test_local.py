import os
import subprocess
import shutil

import pytest

from rampbkd.local import CondaEnvWorker


@pytest.fixture
def get_conda_worker():
    def _create_worker(submission_name):
        module_path = os.path.dirname(__file__)
        ramp_kit_dir = os.path.join(module_path, 'kits', 'iris')
        ramp_data_dir = ramp_kit_dir
        config = {'ramp_kit_dir': os.path.join(module_path, 'kits', 'iris'),
                  'ramp_data_dir': os.path.join(module_path, 'kits', 'iris'),
                  'local_log_folder': os.path.join(
                      module_path, 'kits', 'iris', 'log'),
                  'local_predictions_folder': os.path.join(
                      module_path, 'kits', 'iris', 'predictions'),
                  'conda_env': 'ramp'}
        return CondaEnvWorker(config=config, submission='starting_kit')
    return _create_worker


def test_local_engine(get_conda_worker):
    submission = 'starting_kit'
    worker = get_conda_worker(submission)
    try:
        worker.setup()
        worker.launch_submission()
        print(worker.collect_results())
        worker.teardown()
    finally:
        # remove all directories that we potentially created
        output_training_dir = os.path.join(worker.config['ramp_kit_dir'],
                                           'submissions',
                                           submission,
                                           'training_output')
        log_dir = os.path.join(worker.config['local_log_folder'])
        pred_dir = os.path.join(worker.config['local_predictions_folder'],
                                submission)
        for directory in (output_training_dir,
                          worker.config['local_log_folder'],
                          worker.config['local_predictions_folder']):
            if os.path.exists(directory):
                shutil.rmtree(directory)


# def test_local_engine_unknown_env():
#     module_path = os.path.dirname(__file__)
#     ramp_kit_dir = os.path.join(module_path, 'kits', 'iris')
#     ramp_data_dir = ramp_kit_dir
#     engine = LocalEngine(conda_env='xxx',
#                          ramp_data_dir=ramp_data_dir,
#                          ramp_kit_dir=ramp_kit_dir)
#     msg_err = "The specified conda environment xxx does not exist."
#     with pytest.raises(ValueError, match=msg_err):
#         engine.setup()
