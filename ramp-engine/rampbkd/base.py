import os
import sys
from abc import ABCMeta, abstractmethod


class BaseWorker(metaclass=ABCMeta):
    """Metaclass used to build a RAMP worker. Do not use this class directly.

    Parameters
    ----------
    config : dict
        Configuration of the worker.
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
        self.config = config
        self.submission = submission
        self._status = 'initialized'

    def setup(self):
        self.status = 'setup'

    def teardown(self):
        pass

    @abstractmethod
    def _is_training_finished(self):
        pass

    @property
    def status(self):
        status = self._status
        if status == 'running':
            if self._is_training_finished():
                self._status = 'finished'
        return status

    @status.setter
    def status(self, status):
        self._status = status

    @abstractmethod
    def launch_submission(self):
        pass

    @abstractmethod
    def collect_submission(self):
        pass
