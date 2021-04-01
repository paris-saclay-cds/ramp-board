import logging
import multiprocessing
import numbers
import os
import time

from queue import Queue
from queue import LifoQueue

from ramp_database.tools.submission import get_submissions
from ramp_database.tools.submission import get_submission_by_id
from ramp_database.tools.submission import get_submission_state

from ramp_database.tools.submission import set_bagged_scores
# from ramp_database.tools.submission import set_predictions
from ramp_database.tools.submission import set_time
from ramp_database.tools.submission import set_scores
from ramp_database.tools.submission import set_submission_error_msg
from ramp_database.tools.submission import set_submission_state

from ramp_database.tools.leaderboard import update_all_user_leaderboards
from ramp_database.tools.leaderboard import update_leaderboards
from ramp_database.tools.leaderboard import update_user_leaderboards

from ramp_database.utils import session_scope

from ramp_utils import generate_ramp_config
from ramp_utils import generate_worker_config
from ramp_utils import read_config

from .local import CondaEnvWorker

logger = logging.getLogger('RAMP-DISPATCHER')

log_file = 'dispatcher.log'
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')  # noqa
fileHandler = logging.FileHandler(log_file, mode='a')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)


class Dispatcher:
    """Dispatcher which schedule workers and communicate with the database.

    The dispatcher uses two queues: a queue containing containing the workers
    which should be launched and a queue containing the workers which are being
    processed. The latter queue has a limited size defined by ``n_workers``.
    Note that these workers can run simultaneously.

    Parameters
    ----------
    config : dict or str
        A configuration YAML file containing the information about the
        database.
    event_config : dict or str
        A RAMP configuration YAML file with information regarding the worker
        and the ramp event.
    worker : Worker, default=CondaEnvWorker
        The type of worker to launch. By default, we launch local worker which
        uses ``conda``.
    n_workers : int, default=1
        Maximum number of workers which can run submissions simultaneously.
    n_threads : None or int
        The number of threads that each worker can use. By default, there is no
        limit imposed.
    hunger_policy : {None, 'sleep', 'exit'}
        Policy to apply in case that there is no anymore workers to be
        processed:

        * if None: the dispatcher will work without interruption;
        * if 'sleep': the dispatcher will sleep for 5 seconds before to check
          for new submission;
        * if 'exit': the dispatcher will stop after collecting the results of
          the last submissions.
    time_between_collection : int, default=1
        The amount of time in seconds to wait before checking if we can
        collect results from worker.

        .. note::
           This parameter is important when using a cloud platform to run
           submissions, as the check for collection will be done through SSH.
           Thus, if the time between checks is too small, the repetitive
           SSH requests may be potentially blocked by the cloud provider.
    """
    def __init__(self, config, event_config, worker=None, n_workers=1,
                 n_threads=None, hunger_policy=None,
                 time_between_collection=1):
        self.worker = CondaEnvWorker if worker is None else worker
        self.n_workers = (max(multiprocessing.cpu_count() + 1 + n_workers, 1)
                          if n_workers < 0 else n_workers)
        self.hunger_policy = hunger_policy
        self.time_between_collection = time_between_collection
        # init the poison pill to kill the dispatcher
        self._poison_pill = False
        # create the different dispatcher queues
        self._awaiting_worker_queue = Queue()
        self._processing_worker_queue = LifoQueue(maxsize=self.n_workers)
        self._processed_submission_queue = Queue()
        # split the different configuration required
        if (isinstance(config, str) and
                isinstance(event_config, str)):
            self._database_config = read_config(config,
                                                filter_section='sqlalchemy')
            self._ramp_config = generate_ramp_config(event_config, config)
        else:
            self._database_config = config['sqlalchemy']
            self._ramp_config = event_config['ramp']
        self._worker_config = generate_worker_config(event_config, config)
        # set the number of threads for openmp, openblas, and mkl
        self.n_threads = n_threads
        if self.n_threads is not None:
            if not isinstance(self.n_threads, numbers.Integral):
                raise TypeError(
                    "The parameter 'n_threads' should be a positive integer. "
                    "Got {} instead.".format(repr(self.n_threads))
                    )
            for lib in ('OMP', 'MKL', 'OPENBLAS'):
                os.environ[lib + '_NUM_THREADS'] = str(self.n_threads)
        self._logger = logger.getChild(self._ramp_config['event_name'])

    def fetch_from_db(self, session):
        """Fetch the submission from the database and create the workers."""
        submissions = get_submissions(session,
                                      self._ramp_config['event_name'],
                                      state='new')
        if not submissions:
            return
        for submission_id, submission_name, _ in submissions:
            # do not train the sandbox submission
            submission = get_submission_by_id(session, submission_id)
            if not submission.is_not_sandbox:
                continue
            # create the worker
            worker = self.worker(self._worker_config, submission_name)
            set_submission_state(session, submission_id, 'sent_to_training')
            update_user_leaderboards(
                session, self._ramp_config['event_name'],
                submission .team.name, new_only=True,
            )
            self._awaiting_worker_queue.put_nowait((worker, (submission_id,
                                                             submission_name)))
            self._logger.info(
                f'Submission {submission_name} added to the queue of '
                'submission to be processed'
            )

    def launch_workers(self, session):
        """Launch the awaiting workers if possible."""
        while (not self._processing_worker_queue.full() and
               not self._awaiting_worker_queue.empty()):
            worker, (submission_id, submission_name) = \
                self._awaiting_worker_queue.get()
            self._logger.info(f'Starting worker: {worker}')

            try:
                worker.setup()
                if worker.status != "error":
                    worker.launch_submission()
            except Exception as e:
                self._logger.error(
                    f'Worker finished with unhandled exception:\n {e}'
                )
                worker.status = 'error'
            if worker.status == 'error':
                set_submission_state(session, submission_id, 'checking_error')
                worker.teardown()  # kill the worker
                self._logger.info(
                    f'Worker {worker} killed due to an error '
                    f'while connecting to AWS worker'
                )
                stderr = ("There was a problem with sending your submission"
                          " for training. This problem is on RAMP side"
                          " and most likely it is not related to your"
                          " code. If this happened for the first time"
                          " to this submission you might"
                          " consider submitting the same code once again."
                          " Else, please contact the event organizers."
                          )
                set_submission_error_msg(session, submission_id, stderr)
                continue
            set_submission_state(session, submission_id, 'training')
            submission = get_submission_by_id(session, submission_id)
            update_user_leaderboards(
                session, self._ramp_config['event_name'],
                submission.team.name, new_only=True,
            )
            self._processing_worker_queue.put_nowait(
                (worker, (submission_id, submission_name)))
            self._logger.info(
                f'Store the worker {worker} into the processing queue'
            )

    def collect_result(self, session):
        """Collect result from processed workers."""
        try:
            workers, submissions = zip(
                *[self._processing_worker_queue.get()
                  for _ in range(self._processing_worker_queue.qsize())]
            )
        except ValueError:
            if self.hunger_policy == 'sleep':
                time.sleep(5)
            elif self.hunger_policy == 'exit':
                self._poison_pill = True
            return

        for worker, (submission_id, submission_name) in zip(workers,
                                                            submissions):
            dt = worker.time_since_last_status_check()
            if (dt is not None) and (dt < self.time_between_collection):
                self._processing_worker_queue.put_nowait(
                    (worker, (submission_id, submission_name)))
                time.sleep(0)
                continue
            elif worker.status == 'running':
                self._processing_worker_queue.put_nowait(
                    (worker, (submission_id, submission_name)))
                time.sleep(0)
            elif worker.status == 'retry':
                set_submission_state(session, submission_id, 'new')
                self._logger.info(
                    f'Submission: {submission_id} has been interrupted. '
                    'It will be added to queue again and retried.'
                )
                worker.teardown()
            else:
                self._logger.info(f'Collecting results from worker {worker}')
                returncode, stderr = worker.collect_results()

                if returncode:
                    if returncode == 124:
                        self._logger.info(
                            f'Worker {worker} killed due to timeout.'
                        )
                        submission_status = 'training_error'
                    elif returncode == 2:
                        # Error occurred when downloading the logs
                        submission_status = 'checking_error'
                    else:
                        self._logger.info(
                            f'Worker {worker} killed due to an error '
                            f'during training: {stderr}'
                        )
                        submission_status = 'training_error'
                else:
                    submission_status = 'tested'
                set_submission_state(
                    session, submission_id, submission_status
                )
                set_submission_error_msg(session, submission_id, stderr)
                self._processed_submission_queue.put_nowait(
                    (submission_id, submission_name))
                worker.teardown()

    def update_database_results(self, session):
        """Update the database with the results of ramp_test_submission."""
        make_update_leaderboard = False
        while not self._processed_submission_queue.empty():
            make_update_leaderboard = True
            submission_id, submission_name = \
                self._processed_submission_queue.get_nowait()
            if 'error' in get_submission_state(session, submission_id):
                continue
            self._logger.info(
                f'Write info in database for submission {submission_name}'
            )
            path_predictions = os.path.join(
                self._worker_config['predictions_dir'], submission_name
            )
            # NOTE: In the past we were adding the predictions into the
            # database. Since they require too much space, we stop to store
            # them in the database and instead, keep it onto the disk.
            # set_predictions(session, submission_id, path_predictions)
            set_time(session, submission_id, path_predictions)
            set_scores(session, submission_id, path_predictions)
            set_bagged_scores(session, submission_id, path_predictions)
            set_submission_state(session, submission_id, 'scored')

        if make_update_leaderboard:
            self._logger.info('Update all leaderboards')
            update_leaderboards(session, self._ramp_config['event_name'])
            update_all_user_leaderboards(session,
                                         self._ramp_config['event_name'])
            self._logger.info('Leaderboards updated')

    @staticmethod
    def _reset_submission_after_failure(session, even_name):
        submissions = get_submissions(session, even_name, state=None)
        for submission_id, _, _ in submissions:
            submission_state = get_submission_state(session, submission_id)
            if submission_state in ('training', 'sent_to_training'):
                set_submission_state(session, submission_id, 'new')

    def launch(self):
        """Launch the dispatcher."""
        self._logger.info('Starting the RAMP dispatcher')
        with session_scope(self._database_config) as session:
            self._logger.info('Open a session to the database')
            self._logger.info(
                'Reset unfinished trained submission from previous session'
            )
            self._reset_submission_after_failure(
                session, self._ramp_config['event_name']
            )
            try:
                while not self._poison_pill:
                    self.fetch_from_db(session)
                    self.launch_workers(session)
                    self.collect_result(session)
                    self.update_database_results(session)
            finally:
                # reset the submissions to 'new' in case of error or unfinished
                # training
                self._reset_submission_after_failure(
                    session, self._ramp_config['event_name']
                )
            self._logger.info('Dispatcher killed by the poison pill')
