import logging
import os
import subprocess

from ..base import BaseWorker
from ._conda import CondaEnvWorker

logger = logging.getLogger('RAMP-WORKER')


class DockerWorker(CondaEnvWorker):
    """Local worker which will run a submission within a docker container.

    The worker will run a submission in a docker container. It will use `conda`
    to manage the environment.

    Parameters
    ----------
    config : dict
        Configuration dictionary to set the worker. The following parameter
        should be set:

        * 'conda_env': the name of the conda environment to use. If not
          specified, the base environment will be used.
        * 'kit_dir': path to the directory of the RAMP kit;
        * 'data_dir': path to the directory of the data;
        * 'submissions_dir': path to the directory containing the
          submissions;
        * `logs_dir`: path to the directory where the log of the
          submission will be stored;
        * `predictions_dir`: path to the directory where the
          predictions of the submission will be stored.
        * 'timeout': timeout after a given number of seconds when
          running the worker. If not provided, a default of 7200
          is used.
    submission : str
        Name of the RAMP submission to be handle by the worker.

    Attributes
    ----------
    status : str
        The status of the worker. It should be one of the following state:

            * 'initialized': the worker has been instanciated.
            * 'setup': the worker has been set up.
            * 'running': the worker is training the submission.
            * 'finished': the worker finished to train the submission.
            * 'collected': the results of the training have been collected.
    """
    def setup(self):
        # sanity check for the configuration variable
        for required_param in ('kit_dir', 'data_dir', 'submissions_dir',
                               'logs_dir', 'predictions_dir'):
            self._check_config_name(self.config, required_param)
        self._log_dir = os.path.join(self.config['logs_dir'], self.submission)
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        # get path to conda specified in the path
        conda_path = self.config.get('conda_dir', None)
        docker_image = self.config['docker_image']
        # start the conda image
        docker_cmd = [
            "docker", "run", "-itd", "--rm", "--name", f"{self.submission}"
        ]
        if conda_path is not None:
            # mount the conda path
            docker_cmd += [
                "--mount",
                f'type=bind,source="{conda_path}",target="{conda_path}",readonly'
            ]
            # add it to PATH
            docker_cmd += ["--env", 'PATH="{conda_path}:$PATH"']
        # add ramp-kit directory
        mounted_dir = []
        for key in ["kit_dir", "data_dir", "submissions_dir", "logs_dir"]:
            mount_dir = self.config[key]
            if mount_dir not in mounted_dir:
                mounted_dir.append(mount_dir)
                # docker_cmd += [
                #     "--mount",
                #     r"type=bind,source={},target={}".format(
                #         mount_dir, mount_dir
                #     )
                # ]
                docker_cmd += [
                    "-v", "{}:{}".format(mount_dir, mount_dir)
                ]
        docker_cmd += [f'{docker_image}']
        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if stderr:
            logger.error(stderr.decode())
            raise RuntimeError(stderr.decode())
        self.running_container_hash = stdout.decode().split("\n")[0]
        # find the path to the environment
        self._docker_exec_cmd = ["docker", "exec", "-it"]
        if conda_path is not None:
            # add it to PATH
            self._docker_exec_cmd += [
                "--env", 'PATH="{conda_path}:$PATH"'
            ]
        self._docker_exec_cmd += [
            "--workdir", self.config['kit_dir']
        ]
        self._docker_exec_cmd += ["-u", "root:root"]
        self._docker_exec_cmd += [f"{self.submission}", "/bin/bash", "-c"]
        cmd = self._docker_exec_cmd + ["conda info --envs --json"]
        self._python_bin_path = self._find_conda_env_bin_path(self.config, cmd)
        BaseWorker.setup(self)

    def launch_submission(self):
        print(self._python_bin_path)
        cmd = self._docker_exec_cmd + [
            'ramp-test'
        ]
        self._launch_ramp_test_submission(cmd)
        BaseWorker.launch_submission(self)

    def collect_results(self):
        BaseWorker.collect_results(self)
        if self.status in ['finished', 'running', 'timeout']:
            # communicate() will wait for the process to be completed
            self._proc.communicate()
            self._log_file.close()
            mount_dir = os.path.join(os.getcwd(), self.config["kit_dir"])

    def teardown(self):
        """Remove the predictions stores within the submission."""
        proc = subprocess.run(
            ["docker", "container", "stop", self.running_container_hash],
        )
        super().teardown()
