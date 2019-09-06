import logging
import threading
import queue
import multiprocessing

from datetime import datetime

_LOGGER = logging.getLogger(__name__)

# TODO: This will only work with a single trainer
class LocalJobRunner(object):

    def __init__(self, num_workers = None):
        self.job_id = 0      
        self.jobs = [] 
        self.pending_jobs = queue.Queue()
        self.stop_event = threading.Event()
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
        self.threads = []

    def worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.pending_jobs.get(block=True, timeout=1)
            except queue.Empty:
                continue

            job, target = item
            job_id = job['jobId']
            _LOGGER.info(f"Starting background job '{job_id}'.")
            try:
                job['status'] = 'running'
                job['started'] = datetime.now().isoformat()
                job['result'] = target(job['data'])
                job['status'] = 'completed'
                job['stopped'] = datetime.now().isoformat()
            except Exception as e:
                job['status'] = 'failed'
                job['stopped'] = datetime.now().isoformat()
                job['error'] = str(e)
                _LOGGER.exception('error in worker loop')
            
            self.pending_jobs.task_done()

    def start(self):
        _LOGGER.info(f"local job runner is starting")
        assert(len(self.threads) == 0)     
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker)
            t.deamon = True
            t.start()
            self.threads.append(t)

    def stop(self):
        _LOGGER.info(f"local job runner is stopping")
        self.stop_event.set()
        for t in self.threads:
            t.join(1)
        self.threads = []

    def add_job(self, target, data):
        self.job_id = self.job_id + 1
        job = {
            'jobId': self.job_id,
            'status': 'pending',
            'data': data
        }
        self.jobs.append(job)
        self.pending_jobs.put((job, target))        
        return job
