import logging
import os
import shutil
import subprocess
from datetime import datetime

from ..base import BaseWorker, _get_traceback
from ..conda import _conda_info_envs, _get_conda_env_path

logger = logging.getLogger('RAMP-WORKER')


class RemoteWorker(BaseWorker):
    """Remote dask distributed worker


    This worker uses conda environment to dispatch submission on the
    remote machine. It needs the dask worker to run on the remote machine
    with the same version of dependencies as on the local machine.

    Parameters
    ----------
    config : dict
        Configuration dictionary to set the worker. The following parameter
        should be set:

        * 'conda_env': the name of the *remote* conda environment to use.
        * 'kit_dir': path to the *remote* directory of the RAMP kit;
        * 'data_dir': path to the *remote* directory of the data;
        * 'submissions_dir': path to the *local* directory containing the
          submissions;
        * `logs_dir`: path to the *local* directory where the log of the
          submission will be stored;
        * `predictions_dir`: path to the *local* directory where the
          predictions of the submission will be stored.
        * `dask_scheduler`: URL of the dask scheduler used for submissions.
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
            * 'error': setup failed / training couldn't be started
            * 'running': the worker is training the submission.
            * 'finished': the worker finished to train the submission.
            * 'collected': the results of the training have been collected.
    """
    def __init__(self, config, submission):
        super().__init__(config=config, submission=submission)

    def setup(self):
        """Set up the worker.

        Following steps will be performed;
         - start a dask distributed client
         - find the path to the conda environment to use on the remote machine
        """
        from dask.distributed import Client

        # sanity check for the configuration variable
        for required_param in ('conda_env', 'kit_dir', 'data_dir',
                               'submissions_dir', 'logs_dir',
                               'predictions_dir', 'dask_scheduler'):
            self._check_config_name(self.config, required_param)
        # find the path to the conda environment
        env_name = self.config['conda_env']
        self._client = Client(self.config['dask_scheduler'])

        # Fail early if the Dask worker is not working properly
        self._client.submit(lambda: 1+1).result()

        conda_info = self._client.submit(
                _conda_info_envs, pure=False
        ).result()

        self._python_bin_path = _get_conda_env_path(conda_info, env_name, self)

        super().setup()

    def teardown(self):
        """Remove the predictions stores within the submission."""
        if self.status != 'collected':
            raise ValueError("Collect the results before to kill the worker.")
        output_training_dir = os.path.join(self.config['kit_dir'],
                                           'submissions', self.submission,
                                           'training_output')
        if os.path.exists(output_training_dir):
            shutil.rmtree(output_training_dir)
        super().teardown()

    def _is_submission_finished(self):
        """Status of the submission.

        The submission was launched in a subprocess. Calling ``poll()`` will
        indicate the status of this subprocess.
        """
        self.check_timeout()
        return False if self._proc.poll() is None else True

    def check_timeout(self):
        """Check the submission for timeout."""
        if not hasattr(self, "_start_date"):
            return
        dt = (datetime.utcnow() - self._start_date).total_seconds()
        if dt > self.timeout:
            self._proc.kill()
            self.status = "timeout"
            return True

    @property
    def timeout(self):
        return self.config.get('timeout', 7200)

    def launch_submission(self):
        """Launch the submission.

        Basically, it comes to run ``ramp_test_submission`` using the conda
        environment given in the configuration. The submission is launched in
        a subprocess to free to not lock the Python main process.
        """
        cmd_ramp = os.path.join(self._python_bin_path, 'ramp-test')
        if self.status == 'running':
            raise ValueError('Wait that the submission is processed before to '
                             'launch a new one.')
        self._log_dir = os.path.join(self.config['logs_dir'], self.submission)
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        self._log_file = open(os.path.join(self._log_dir, 'log'), 'wb+')
        self._proc = subprocess.Popen(
            [cmd_ramp,
             '--submission', self.submission,
             '--ramp-kit-dir', self.config['kit_dir'],
             '--ramp-data-dir', self.config['data_dir'],
             '--ramp-submission-dir', self.config['submissions_dir'],
             '--save-output',
             '--ignore-warning'],
            stdout=self._log_file,
            stderr=self._log_file,
        )
        super().launch_submission()
        self._start_date = datetime.utcnow()

    def collect_results(self):
        """Collect the results after that the submission is completed.

        Be aware that calling ``collect_results()`` before that the submission
        finished will lock the Python main process awaiting for the submission
        to be processed. Use ``worker.status`` to know the status of the worker
        beforehand.
        """
        super().collect_results()
        if self.status in ['finished', 'running', 'timeout']:
            # communicate() will wait for the process to be completed
            self._proc.communicate()
            self._log_file.close()
            with open(os.path.join(self._log_dir, 'log'), 'rb') as f:
                log_output = f.read()
            error_msg = _get_traceback(log_output.decode('utf-8'))
            if self.status == 'timeout':
                error_msg += ('\nWorker killed due to timeout after {}s.'
                              .format(self.timeout))
            if self.status == 'timeout':
                returncode = 124
            else:
                returncode = self._proc.returncode
            pred_dir = os.path.join(
                self.config['predictions_dir'], self.submission
            )
            output_training_dir = os.path.join(
                self.config['submissions_dir'], self.submission,
                'training_output')
            if os.path.exists(pred_dir):
                shutil.rmtree(pred_dir)
            if returncode:
                if os.path.exists(output_training_dir):
                    shutil.rmtree(output_training_dir)
                self.status = 'collected'
                return (returncode, error_msg)
            # copy the predictions into the disk
            # no need to create the directory, it will be handle by copytree
            shutil.copytree(output_training_dir, pred_dir)
            self.status = 'collected'
            return (returncode, error_msg)
