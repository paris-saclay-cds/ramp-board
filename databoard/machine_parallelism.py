from multiprocessing.managers import BaseManager
import sys
import Queue
import uuid
import os
import time
import traceback
# from sklearn.externals.joblib import Parallel, delayed


sys.path.append(os.path.dirname(__file__) + "/..")
"""
First, you need to run a server
> python machine_parallelism.py server
Then one or several clients on machines that can access the server
(host, port and password are fixed below)
> python machine_parallism.py client

These are the machines (the client ones) that  will
launch jobs (training models).

Make sure that machine parallelism is activated in
databoard/train_test.py : both is_parallelize and
and parallelize_across_machines must be set to True.

Now, you can train with machine parallelism :

> fab train

for instance.

## IMPORTANT

for this to work well with databoard,
make sure you run machine_parallelism.py in the root
directory of the ramp, where ramp_index.txt is.

so you should run

> python databoard/machine_parallelism.py client --host=serverhost

for the clients

and

> python databoard/machine_parallelism.py server

for the server

"""


class QueueManager(BaseManager):
    pass


class TimeoutError(OSError):
    pass

host, port, password = '0.0.0.0', 50000, 'insects'


def register_queues():
    available_jobs = Queue.Queue()
    QueueManager.register('get_available_jobs', callable=lambda: available_jobs)
    finished_jobs = Queue.Queue()
    QueueManager.register('get_finished_jobs', callable=lambda: finished_jobs)


def build_queue_manager(host, port):
    q = QueueManager(address=(host, port), authkey=password)
    print("Using host={0} and port={1}".format(host, port))
    return q


def serve_forever(host=host, port=port):
    register_queues()
    q = build_queue_manager(host, port)
    s = q.get_server()
    s.serve_forever()


class Logger(object):
    def __init__(self, fd, terminal):
        self.terminal = terminal
        self.log = fd

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def run_job(job_params, output_folder="."):
    title, job_id, method, args = job_params
    print("Got a new job : {0}, let's run it !".format(title))
    filename = os.path.join(output_folder, title + ".out")
    try:
        fd = open(filename, "w", buffering=1)
        bak_stdout = sys.stdout
        bak_stderr = sys.stderr
        sys.stdout = Logger(fd, sys.stdout)
        sys.stderr = sys.stdout
        status = method(*args)
        fd.close()
        sys.stdout = bak_stdout
        sys.stderr = bak_stderr
    except Exception, e:
        print("Exception : ", repr(e))
        trback = traceback.format_exc()
        print(trback)
        e.traceback = trback
        status = e
    return status


def be_client_forever(host=host, port=port, output_folder="."):
    register_queues()
    q = build_queue_manager(host=host, port=port)
    q.connect()
    available_jobs = q.get_available_jobs()
    finished_jobs = q.get_finished_jobs()
    while True:
        job_params = available_jobs.get()
        _, job_id, _, _ = job_params
        # run it as a separate process
        #status = (Parallel(n_jobs=1, backend='multiprocessing')( [ delayed(run_job)(job_params, output_folder) ] ))[0]
        status = run_job(job_params, output_folder)
        finished_jobs.put((job_id, status))

def put_job(method, args, host=host, port=port, title=""):
    register_queues()
    job_id = str(uuid.uuid1())
    q = build_queue_manager(host=host, port=port)
    q.connect()
    available_jobs = q.get_available_jobs()
    available_jobs.put((title, job_id, method, args))
    print("Putting a new job : {0} with method {1}".format(job_id, method))
    return job_id


def wait_for(job_ids, host=host, port=port, timeout=10000):
    t = time.time()
    job_status = dict()
    register_queues()
    q = build_queue_manager(host=host, port=port)
    q.connect()
    job_ids = set(job_ids.copy())
    finished_jobs = q.get_finished_jobs()
    while len(job_ids) > 0:
        job_id, status = finished_jobs.get()
        if job_id in job_ids:
            job_status[job_id] = status
            job_ids.remove(job_id)
            print("Job {0} finished".format(job_id))
        else:
            finished_jobs.put((job_id, status))

        if len(job_ids) > 0 and int(time.time() - t) > timeout:
            raise TimeoutError()

    return job_status

wait_for_jobs_and_get_status = wait_for

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="client|server")
    parser.add_argument("--host", help="host", default=host)
    parser.add_argument("--port", help="port", default=port, type=int)
    parser.add_argument("--output-folder", help="output-folder", default=".", type=str)

    args = parser.parse_args()
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)
    if args.mode == "client":
        be_client_forever(host=args.host, port=args.port, output_folder=args.output_folder)
    elif args.mode == "server":
        serve_forever(host=args.host, port=args.port)
    else:
        print("Unkown command, please use either client or server")
    sys.exit(0)
