import os
import subprocess
import time

from .base import BaseEngine


class LocalEngine(BaseEngine):
    def __init__(self, conda_env='base', submission='starting_kit',
                 ramp_kit_dir='.', ramp_data_dir='.'):
        self._log = {}
        return super().__init__(conda_env=conda_env,
                                submission=submission,
                                ramp_kit_dir=ramp_kit_dir,
                                ramp_data_dir=ramp_data_dir)

    def setup(self):
        self.status = 'setup'
        return super().setup()

    def teardown(self):
        return super().teardown()

    def _is_training_finished(self):
        return False if self._process_submission.poll() is None else True

    def launch_submission(self):
        cmd_ramp = os.path.join(self._python_bin_path,
                                'ramp_test_submission')
        if self.status == 'running':
            raise ValueError('Wait that the submission is processed before to '
                             'launch a new one.')
        self._process_submission = subprocess.Popen(
            [cmd_ramp,
             '--submission', self.submission,
             '--ramp_kit_dir', self.ramp_kit_dir,
             '--ramp_data_dir', self.ramp_data_dir,
             '--save-y-preds'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.status = 'running'

    def collect_submission(self):
        if self.status == 'finished':
            if self._process_submission.pid not in self._log:
                log, _ = self._process_submission.communicate()
                self._log[self._process_submission.pid] = log
                self.logger.error('The training of the submission "{}" failed.'
                                  .format(self.submission))
            return self._log[self._process_submission.pid]
