import os
import sys
from abc import ABCMeta, abstractmethod
import six


class BaseWorker(six.with_metaclass(ABCMeta)):
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
        self.status = 'initialized'

    def setup(self):
        """Setup the worker with some given setting required before launching
        a submission."""
        self.status = 'setup'

    def teardown(self):
        """Clean up (i.e., removing path, etc.) before killing the worker."""
        pass

    @abstractmethod
    def _is_submission_finished(self):
        """Indicate the status of submission"""
        pass

    @property
    def status(self):
        status = self._status
        if status == 'running':
            if self._is_submission_finished():
                self._status = 'finished'
        return status

    @status.setter
    def status(self, status):
        self._status = status

    @abstractmethod
    def launch_submission(self):
        """Launch a submission to be trained."""
        pass

    @abstractmethod
    def collect_results(self):
        """Collect the results after submission training."""
        if self.status == 'initialized':
            raise ValueError('The worker has not been setup and no submission '
                             'was launched. Call the method setup() and '
                             'launch_submission() before to collect the '
                             'results.')
        elif self.status == 'setup':
            raise ValueError('No submission was launched. Call the method '
                             'launch_submission() and then try again to '
                             'collect the results.')
