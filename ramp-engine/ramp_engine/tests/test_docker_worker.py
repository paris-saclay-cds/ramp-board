import os
import shutil

import pytest

from ramp_engine.local import DockerWorker


@pytest.fixture
def get_docker_worker(submission):
    def _create_worker(
        submission_name, conda_env='base',
        docker_image="continuumio/miniconda3",
    ):
        module_path = os.path.dirname(__file__)
        config = {
            'kit_dir': os.path.join(module_path, 'kits', 'iris'),
            'data_dir': os.path.join(module_path, 'kits', 'iris'),
            'submissions_dir': os.path.join(module_path, 'kits',
                                            'iris', 'submissions'),
            'logs_dir': os.path.join(module_path, 'kits', 'iris', 'log'),
            'predictions_dir': os.path.join(
                      module_path, 'kits', 'iris', 'predictions'
            ),
            'conda_env': conda_env,
            'docker_image': docker_image,
        }
        return DockerWorker(config=config, submission=submission)
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
def test_docker_worker(submission, get_docker_worker):
    worker = get_docker_worker(submission)
    try:
        assert worker.status == 'initialized'
        worker.setup()
        assert worker.status == 'setup'
        worker.launch_submission()
        worker.collect_results()
    finally:
        # remove all directories that we potentially created
        worker._status = 'collected'
        worker.teardown()
        # _remove_directory(worker)
