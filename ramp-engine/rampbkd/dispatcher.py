
import logging
import multiprocessing
import os
import sys
import time

PYTHON_MAJOR_VERSION = sys.version_info[0]
if PYTHON_MAJOR_VERSION >= 3:
    from queue import Queue
    from queue import LifoQueue
else:
    from Queue import Queue
    from Queue import LifoQueue

from databoard import ramp_config
# from databoard.db_tools import get_submissions
from databoard.db_tools import get_submission_on_cv_folds
from databoard.db_tools import update_all_user_leaderboards
from databoard.db_tools import update_leaderboards
from databoard.db_tools import update_submission_on_cv_fold

from rampdb.tools import get_submissions
from rampdb.tools import get_submission_state
from rampdb.tools import set_submission_state

from ramputils import generate_worker_config

from .local import CondaEnvWorker

logger = logging.getLogger('DISPATCHER')


class Dispatcher(object):
    """Dispatcher which schedule workers and communicate with the database.

    The dispatcher uses two queues: a queue containing containing the workers
    which should be launched and a queue containing the workers which are being
    processed. The latter queue has a limited size defined by ``n_workers``.
    Note that these workers can run simultaneously.

    Parameters
    ----------
    config : dict,
        A RAMP configuration YAML file with information regarding the worker
        and the ramp event.
    worker : Worker, default=CondaEnvWorker
        The type of worker to launch. By default, we launch local worker which
        uses ``conda``.
    n_workers : int, default=1
        Maximum number of workers which can run submissions simultaneously.
    worker_policy : {None, 'sleep', 'exit'}
        Policy to apply in case that there is no current workers processed.
    """
    def __init__(self, config, worker=None, n_worker=1, worker_policy=None):
        self.config = config
        self.worker = CondaEnvWorker if worker is None else worker
        self.n_worker = (max(multiprocessing.cpu_count() + 1 + n_worker, 1)
                         if n_worker < 0 else n_worker)
        self.worker_policy = worker_policy
        self._poison_pill = False
        self._awaiting_worker_queue = Queue()
        self._processing_worker_queue = LifoQueue(maxsize=self.n_worker)
        self._processed_submission_queue = Queue()
        self._worker_config = generate_worker_config(self.config)

    def fetch_from_db(self):
        """Fetch the submission from the database and create the workers."""
        submissions = get_submissions(self.config['sqlalchemy'],
                                      self.config['ramp']['event_name'],
                                      state='new')
        if not submissions:
            logger.info('No new submissions fetch from the database')
            return
        for submission_id, submission_name, _ in submissions:
            # create the worker
            worker = self.worker(self._worker_config, submission_name)
            set_submission_state(self.config['sqlalchemy'], submission_id,
                                 'sent_to_training')
            self._awaiting_worker_queue.put_nowait((worker, (submission_id,
                                                             submission_name)))
            logger.info('Submission {} added to the queue of submission to be '
                        'processed'.format(submission_name))

    def launch_workers(self):
        """Launch the awaiting workers if possible."""
        while (not self._processing_worker_queue.full() and
               not self._awaiting_worker_queue.empty()):
            worker, (submission_id, submission_name) = \
                self._awaiting_worker_queue.get()
            logger.info('Starting worker: {}'.format(worker))
            worker.setup()
            worker.launch_submission()
            set_submission_state(self.config['sqlalchemy'], submission_id,
                                 'training')
            self._processing_worker_queue.put_nowait(
                (worker, (submission_id, submission_name)))
            logger.info('Store the worker {} into the processing queue'
                        .format(worker))
        if self._processing_worker_queue.full():
            logger.info('The processing queue is full. Waiting for a worker to'
                        ' finish')

    def collect_result(self):
        """Collect result from processed workers."""
        try:
            workers, submissions = zip(
                *[self._processing_worker_queue.get()
                  for _ in range(self._processing_worker_queue.qsize())]
            )
        except ValueError:
            logger.info('No workers are currently waiting or processed.')
            if self.worker_policy == 'sleep':
                time.sleep(5)
            elif self.worker_policy == 'exit':
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
                returncode = worker.collect_results()
                set_submission_state(
                    self.config['sqlalchemy'], submission_id,
                    'trained' if not returncode else 'training_error'
                )
                self._processed_submission_queue.put_nowait(
                    (submission_id, submission_name))
                worker.teardown()

    def update_database_results(self):
        """Update the database with the results of ramp_test_submission."""
        while not self._processed_submission_queue.empty():
            submission_id, submission_name = \
                self._processed_submission_queue.get_nowait()
            if 'error' in get_submission_state(self.config['sqlalchemy'],
                                               submission_id):
                # do not make any update in case of failed submission
                logger.info('Skip update for {} due to failure during the '
                            'processing'.format(submission_name))
                continue
            logger.info('Update the results obtained on each fold for '
                        '{}'.format(submission_name))
            submission_cv_folds = get_submission_on_cv_folds(submission_id)
            for fold_idx, sub_cv_fold in enumerate(submission_cv_folds):
                path_results = os.path.join(
                    self._worker_config['predictions_dir'],
                    submission_name, 'fold_{}'.format(fold_idx)
                )
                update_submission_on_cv_fold(sub_cv_fold, path_results)
                # TODO: test those two last functions
                update_leaderboards(self.config['ramp']['event_name'])
                update_all_user_leaderboards(self.config['ramp']['event_name'])

    def launch(self):
        """Launch the dispatcher."""
        logger.info('Starting the RAMP dispatcher')
        # try:
        while not self._poison_pill:
            self.fetch_from_db()
            self.launch_workers()
            self.collect_result()
            self.update_database_results()
        # finally:
        #     # reset the submissions to 'new' in case of error or unfinished
        #     # training
        #     submissions = get_submissions(event_name=self.config['event_name'])
        #     for submission in submissions:
        #         if 'training' in submission.state:
        #             submission.state = 'new'
        logger.info('Dispatcher killed by the poison pill')
