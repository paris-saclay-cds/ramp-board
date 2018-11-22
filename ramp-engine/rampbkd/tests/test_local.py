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
                  'conda_env': 'ramp'}
        return CondaEnvWorker(config=config, submission='starting_kit')
    return _create_worker


def test_local_engine(get_conda_worker):
    worker = get_conda_worker('starting_kit')
    try:
        worker.setup()
        worker.launch_submission()
        worker.collect_results()
        worker.teardown()
    finally:
        output_training = os.path.join(worker.config['ramp_kit_dir'],
                                       'submissions',
                                       'starting_kit',
                                       'training_output')
        if os.path.exists(output_training):
            shutil.rmtree(output_training)


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
