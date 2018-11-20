import json
import os
import subprocess
from abc import ABCMeta, abstractmethod

from rampwf.utils.testing import assert_submission


class BaseEngine(metaclass=ABCMeta):
    def __init__(self, conda_env='base', submission='starting_kit',
                 ramp_kit_dir='.', ramp_data_dir='.'):
        self.conda_env = conda_env
        self.submission = submission
        self.ramp_kit_dir = ramp_data_dir
        self.ramp_data_dir = ramp_data_dir

    def _find_conda_environment(self):
        command_line_process = subprocess.Popen(
            ["conda", "info", "--envs", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        process_output, _ = command_line_process.communicate()
        conda_info = json.loads(process_output)

        if self.conda_env == 'base':
            self._python_bin_path = os.path.join(conda_info['envs'][0], 'bin')
        else:
            envs_path = conda_info['envs'][1:]
            if not envs_path:
                raise ValueError('Only the base environment exist. You need '
                                 'to create the {} conda environment to use '
                                 'it.'.format(self.conda_env))
            is_env_found = False
            for env in envs_path:
                if self.conda_env == os.path.split(env)[-1]:
                    is_env_found = True
                    self._python_bin_path = os.path.join(env, 'bin')
                    break
            if not is_env_found:
                raise ValueError('The specified conda environment {} does not '
                                 'exist. You need to create it.'
                                 .format(self.conda_env))

    @abstractmethod
    def setup(self):
        self._find_conda_environment()

    @abstractmethod
    def teardown(self):
        pass

    @property
    @abstractmethod
    def status(self):
        pass

    @abstractmethod
    def launch_submission(self):
        pass

    @abstractmethod
    def collect_submission(self):
        pass
