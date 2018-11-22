import json
import os
import subprocess

from .base import BaseWorker


class CondaEnvWorker(BaseWorker):
    """Local worker which uses conda environment to dispatch submission.

    Parameters
    ----------
    config : dict
        Configuration dictionary to set the worker. The following parameter
        should be set:

        * 'conda_env': the name of the conda environment to use. If not
          specified, the base environment will be used.
        * 'ramp_kit_dir': path to the directory of the RAMP kit;
        * 'ramp_data_dir': path to the directory of the data.
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
    def __init__(self, config, submission):
        super(CondaEnvWorker, self).__init__(config=config,
                                             submission=submission)

    def setup(self):
        """Set up the worker.

        The worker will find the path to the conda environment to use using
        the configuration passed when instantiating the worker.
        """
        # find the path to the conda environment
        env_name = (self.config['conda_env']
                    if 'conda_env' in self.config.keys() else 'base')
        proc = subprocess.Popen(
            ["conda", "info", "--envs", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        stdout, _ = proc.communicate()
        conda_info = json.loads(stdout)

        if env_name == 'base':
            self._python_bin_path = os.path.join(conda_info['envs'][0], 'bin')
        else:
            envs_path = conda_info['envs'][1:]
            if not envs_path:
                raise ValueError('Only the conda base environment exist. You '
                                 'need to create the "{}" conda environment '
                                 'to use it.'.format(env_name))
            is_env_found = False
            for env in envs_path:
                if env_name == os.path.split(env)[-1]:
                    is_env_found = True
                    self._python_bin_path = os.path.join(env, 'bin')
                    break
            if not is_env_found:
                raise ValueError('The specified conda environment {} does not '
                                 'exist. You need to create it.'
                                 .format(env_name))

    def _is_submission_finished(self):
        """Status of the submission.

        The submission was launched in a subprocess. Calling ``poll()`` will
        indicate the status of this subproces.
        """
        return False if self._proc.poll() is None else True

    def launch_submission(self):
        """Launch the submission.

        Basically, it comes to run ``ramp_test_submission`` using the conda
        environment given in the configuration. The submission is launched in
        a subprocess to free to not lock the Python main process.
        """
        cmd_ramp = os.path.join(self._python_bin_path, 'ramp_test_submission')
        if self.status == 'running':
            raise ValueError('Wait that the submission is processed before to '
                             'launch a new one.')
        self._proc = subprocess.Popen(
            [cmd_ramp,
             '--submission', self.submission,
             '--ramp_kit_dir', self.config['ramp_kit_dir'],
             '--ramp_data_dir', self.config['ramp_data_dir'],
             '--save-y-preds'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.status = 'running'

    def collect_results(self):
        """Collect the results after that the submission is completed.

        Be aware that calling ``collect_results()`` before that the submission
        finished will lock the Python main process awaiting for the submission
        to be processed. Use ``worker.status`` to know the status of the worker
        beforehand.
        """
        if self.status == 'finished' or self.status == 'running':
            # communicate() will wait for the process to be completed.
            self._proc_log, _ = self._proc.communicate()
            self.status = 'collected'
            return self._proc_log
        elif self.status == 'collected':
            return self._proc_log
