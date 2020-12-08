import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime
import subprocess

logger = logging.getLogger('RAMP-WORKER')

log_file = "worker.log"
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')  # noqa
fileHandler = logging.FileHandler(log_file, mode='a')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)


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

            * 'initialized': the worker has been instantiated.
            * 'setup': the worker has been set up.
            * 'error': setup failed / training couldn't be started.
            * 'running': the worker is training the submission.
            * 'finished': the worker finished to train the submission.
            * 'collected': the results of the training have been collected.
            * 'retry': the worker has been interrupted (and will be retried).
            * 'killed'
    """
    def __init__(self, config, submission):
        self.config = config
        self.submission = submission
        self.status = 'initialized'

    def setup(self):
        """Setup the worker with some given setting required before launching
        a submission."""
        self.status = 'setup'

    @staticmethod
    def _check_config_name(config, param):
        if param not in config.keys():
            raise ValueError("The worker required the parameter '{}' in the "
                             "configuration given at instantiation. Only {}"
                             "parameters were given."
                             .format(param, config.keys()))

    def teardown(self):
        """Clean up (i.e., removing path, etc.) before killing the worker."""
        self.status = 'killed'

    @abstractmethod
    def _is_submission_interrupted(self):
        """Check if submission has been interrupted."""
        pass

    @abstractmethod
    def _is_submission_finished(self):
        """Indicate the status of submission"""
        pass

    @property
    def status(self):
        status = self._status
        try:
            if status == 'running':
                self._status_running_check_time = datetime.utcnow()
                if self._is_submission_interrupted():
                    self._status = 'retry'
                elif self._is_submission_finished():
                    self._status = 'finished'
        except subprocess.CalledProcessError:
            # there was a problem while connecting to the worker
            # if you are using AWS it might be that an instance was terminated
            # from outside. retry the submission
            self._status = 'retry'
        return self._status

    @status.setter
    def status(self, status):
        self._status = status

    def check_timeout(self):
        """Check a submission for timeout."""
        pass

    def time_since_last_status_check(self):
        """Calculate time elapsed since the last 'running' status check, in
        seconds.

        Returns
        -------
        elapsed_time : int or None
            Time elapsed in seconds since the last 'running' status
            check. If `_status_running_check_time` attribute does not
            exist (e.g. `worker.status` has not been called once status
            is 'running'), `None` is returned.
        """
        if not hasattr(self, "_status_running_check_time"):
            return None
        elapsed_time = ((datetime.utcnow() -
                        self._status_running_check_time).total_seconds())
        return elapsed_time

    @abstractmethod
    def launch_submission(self):
        """Launch a submission to be trained."""
        self.status = 'running'

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

    def launch(self):
        """Launch a standalone RAMP worker.

        You can use this method when you want to use a worker without using
        the RAMP dispatcher.
        """
        self.setup()
        self.launch_submission()
        # collecting the results will block the process until the submission
        # is processed
        self.collect_results()
        self.teardown()

    def __str__(self):
        msg = ('{worker_name}({submission_name}): status="{status}"'
               .format(worker_name=self.__class__.__name__,
                       submission_name=self.submission,
                       status=self.status))
        return msg

    def __repr__(self):
        return self.__str__()


def _get_traceback(content):
    """
    Get the traceback part from the content containing the standard
    error/output of a python process. It is used to get the traceback
    of `ramp_test_submission` when there is an error.

    Parameters
    ----------
    content : str

    Returns
    -------
    str with the traceback

    """
    if not content:
        return ''
    # cut_exception_text = content.rfind('--->')
    # was like commented line above in ramp-board
    # but there is no ---> in logs when we use
    # ramp_test_submission, so we just search for the
    # first occurence of 'Traceback'.
    cut_exception_text = content.find('Traceback')
    if cut_exception_text > 0:
        content = content[cut_exception_text:]
    return content
