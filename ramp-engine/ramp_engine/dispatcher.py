import logging
import multiprocessing
import os
import time

from queue import Queue
from queue import LifoQueue

from ramp_database.tools.submission import get_submissions
from ramp_database.tools.submission import get_submission_by_id
from ramp_database.tools.submission import get_submission_state

from ramp_database.tools.submission import set_bagged_scores
from ramp_database.tools.submission import set_predictions
from ramp_database.tools.submission import set_time
from ramp_database.tools.submission import set_scores
from ramp_database.tools.submission import set_submission_error_msg
from ramp_database.tools.submission import set_submission_state

from ramp_database.tools.leaderboard import update_all_user_leaderboards
from ramp_database.tools.leaderboard import update_leaderboards

from ramp_database.utils import session_scope

from ramp_utils import generate_ramp_config
from ramp_utils import generate_worker_config
from ramp_utils import read_config

from .local import CondaEnvWorker

logger = logging.getLogger('RAMP-DISPATCHER')


class Dispatcher:
    """Dispatcher which schedule workers and communicate with the database.

    The dispatcher uses two queues: a queue containing containing the workers
    which should be launched and a queue containing the workers which are being
    processed. The latter queue has a limited size defined by ``n_workers``.
    Note that these workers can run simultaneously.

    Parameters
    ----------
    config : dict or str
        A configuration YAML file containing the inforation about the database.
    event_config : dict or str
        A RAMP configuration YAML file with information regarding the worker
        and the ramp event.
    worker : Worker, default=CondaEnvWorker
        The type of worker to launch. By default, we launch local worker which
        uses ``conda``.
    n_workers : int, default=1
        Maximum number of workers which can run submissions simultaneously.
    hunger_policy : {None, 'sleep', 'exit'}
        Policy to apply in case that there is no anymore workers to be
        processed:

        * if None: the dispatcher will work without interruption;
        * if 'sleep': the dispatcher will sleep for 5 seconds before to check
          for new submission;
        * if 'exit': the dispatcher will stop after collecting the results of
          the last submissions.
    """
    def __init__(self, config, event_config, worker=None, n_worker=1,
                 hunger_policy=None):
        self.worker = CondaEnvWorker if worker is None else worker
        self.n_worker = (max(multiprocessing.cpu_count() + 1 + n_worker, 1)
                         if n_worker < 0 else n_worker)
        self.hunger_policy = hunger_policy
        # init the poison pill to kill the dispatcher
        self._poison_pill = False
        # create the different dispatcher queues
        self._awaiting_worker_queue = Queue()
        self._processing_worker_queue = LifoQueue(maxsize=self.n_worker)
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

    def fetch_from_db(self, session):
        """Fetch the submission from the database and create the workers."""
        submissions = get_submissions(session,
                                      self._ramp_config['event_name'],
                                      state='new')
        if not submissions:
            logger.info('No new submissions fetch from the database')
            return
        for submission_id, submission_name, _ in submissions:
            # do not train the sandbox submission
            submission = get_submission_by_id(session, submission_id)
            if not submission.is_not_sandbox:
                continue
            # create the worker
            worker = self.worker(self._worker_config, submission_name)
            set_submission_state(session, submission_id, 'sent_to_training')
            self._awaiting_worker_queue.put_nowait((worker, (submission_id,
                                                             submission_name)))
            logger.info('Submission {} added to the queue of submission to be '
                        'processed'.format(submission_name))

    def launch_workers(self, session):
        """Launch the awaiting workers if possible."""
        while (not self._processing_worker_queue.full() and
               not self._awaiting_worker_queue.empty()):
            worker, (submission_id, submission_name) = \
                self._awaiting_worker_queue.get()
            logger.info('Starting worker: {}'.format(worker))
            worker.setup()
            worker.launch_submission()
            set_submission_state(session, submission_id, 'training')
            self._processing_worker_queue.put_nowait(
                (worker, (submission_id, submission_name)))
            logger.info('Store the worker {} into the processing queue'
                        .format(worker))
        if self._processing_worker_queue.full():
            logger.info('The processing queue is full. Waiting for a worker to'
                        ' finish')

    def collect_result(self, session):
        """Collect result from processed workers."""
        try:
            workers, submissions = zip(
                *[self._processing_worker_queue.get()
                  for _ in range(self._processing_worker_queue.qsize())]
            )
        except ValueError:
            logger.info('No workers are currently waiting or processed.')
            if self.hunger_policy == 'sleep':
                time.sleep(5)
            elif self.hunger_policy == 'exit':
                self._poison_pill = True
            return
        for worker, (submission_id, submission_name) in zip(workers,
                                                            submissions):
            if worker.status == 'running':
                self._processing_worker_queue.put_nowait(
                    (worker, (submission_id, submission_name)))
                logger.info('Worker {} is still running'.format(worker))
                time.sleep(0)
            else:
                logger.info('Collecting results from worker {}'.format(worker))
                returncode, stderr = worker.collect_results()
                set_submission_state(
                    session, submission_id,
                    'tested' if not returncode else 'training_error'
                )
                set_submission_error_msg(session, submission_id, stderr)
                self._processed_submission_queue.put_nowait(
                    (submission_id, submission_name))
                worker.teardown()

    def update_database_results(self, session):
        """Update the database with the results of ramp_test_submission."""
        while not self._processed_submission_queue.empty():
            submission_id, submission_name = \
                self._processed_submission_queue.get_nowait()
            if 'error' in get_submission_state(session, submission_id):
                update_leaderboards(session, self._ramp_config['event_name'])
                update_all_user_leaderboards(session,
                                             self._ramp_config['event_name'])
                logger.info('Skip update for {} due to failure during the '
                            'processing'.format(submission_name))
                continue
            logger.info('Update the results obtained on each fold for '
                        '{}'.format(submission_name))
            path_predictions = os.path.join(
                self._worker_config['predictions_dir'], submission_name
            )
            set_predictions(session, submission_id, path_predictions)
            set_time(session, submission_id, path_predictions)
            set_scores(session, submission_id, path_predictions)
            set_bagged_scores(session, submission_id, path_predictions)
            set_submission_state(session, submission_id, 'scored')
            update_leaderboards(session, self._ramp_config['event_name'])
            update_all_user_leaderboards(session,
                                         self._ramp_config['event_name'])

    def launch(self):
        """Launch the dispatcher."""
        logger.info('Starting the RAMP dispatcher')
        with session_scope(self._database_config) as session:
            logger.info('Open a session to the database')
            try:
                while not self._poison_pill:
                    self.fetch_from_db(session)
                    self.launch_workers(session)
                    self.collect_result(session)
                    self.update_database_results(session)
            finally:
                # reset the submissions to 'new' in case of error or unfinished
                # training
                submissions = get_submissions(session,
                                              self._ramp_config['event_name'],
                                              state=None)
                for submission_id, _, _ in submissions:
                    submission_state = get_submission_state(session,
                                                            submission_id)
                    if submission_state in ('training', 'send_to_training'):
                        set_submission_state(session, submission_id, 'new')
            logger.info('Dispatcher killed by the poison pill')
