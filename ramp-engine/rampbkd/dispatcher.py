import sys

PYTHON_MAJOR_VERSION = sys.version_info[0]
if PYTHON_MAJOR_VERSION >= 3:
    from queue import Queue
    from queue import LifoQueue
else:
    from Queue import Queue
    from Queue import LifoQueue

from databoard import ramp_config
from databoard.db_tools import get_submissions

from .local import CondaEnvWorker


class Dispatcher:
    """Dispatcher which schedule workers and communicate with the database.

    The dispatcher uses two queues: a queue containing containing the workers
    which should be launched and a queue containing the workers which are being
    processed. The latter queue has a limited size defined by ``n_workers``.
    Note that these workers can run simultaneously.

    Parameters
    ----------
    config : dict,
        A dictionary with all necessary configuration.
    worker : Worker, default=CondaEnvWorker
        The type of worker to launch. By default, we launch local worker which
        uses ``conda``.
    n_workers : int, default=1
        Maximum number of workers which can run submissions simultaneously.
    """
    def __init__(self, config, worker=None, n_worker=1):
        self.config = config
        self.worker = CondaEnvWorker if worker is None else worker
        self.n_worker = n_worker
        self._awaiting_worker_queue = Queue()
        self._processing_worker_queue = LifoQueue(maxsize=self.n_worker)

    def fetch_from_db(self):
        """Fetch the submission from the database and create the workers."""
        submissions = get_submissions(event_name=self.config['event_name'])
        for sub in submissions:
            # TODO: need to change the submission name
            # TODO: modify ramp-workflow such that we can specify the path of
            # the submission directory
            worker = self.worker(config, sub.name)
            self._awaiting_worker_queue.put_nowait(worker)
        # if generated_submission is not None:
        #     # temporary path to the submissions
        #     module_path = os.path.dirname(__file__)
        #     config = {'ramp_kit_dir': os.path.join(
        #                     module_path, 'kits', 'iris'),
        #                 'ramp_data_dir': os.path.join(
        #                     module_path, 'kits', 'iris'),
        #                 'local_log_folder': os.path.join(
        #                     module_path, 'kits', 'iris', 'log'),
        #                 'local_predictions_folder': os.path.join(
        #                     module_path, 'kits', 'iris', 'predictions'),
        #                 'conda_env': 'ramp'}
        #     worker = CondaEnvWorker(config, generated_submission)
        #     await self._awaiting_worker_queue.put(worker)

    # async def launch_worker(self):
    #     """Coroutine to launch awaiting workers."""
    #     while True:
    #         worker = await self._awaiting_worker_queue.get()
    #         print(f'launch worker {worker}')
    #         worker.setup()
    #         await worker.launch_submission()
    #         await self._processing_worker_queue.put(worker)
    #         print(f'queue worker {worker}')

    # async def collect_result(self):
    #     """Collect result from processed workers."""
    #     while True:
    #         worker = await self._processing_worker_queue.get()
    #         if worker.status == 'running':
    #             # await process_queue.put(proc) lock proc.returncode to change
    #             # status.
    #             self._processing_worker_queue.put_nowait(worker)
    #             await asyncio.sleep(0)
    #         else:
    #             print(f'collect results of worker {worker}')
    #             await worker.collect_results()
    #             worker.teardown()