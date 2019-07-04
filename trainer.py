import sys
import os
import logging
import logging.config

import threading
import queue

from vehicletracker.data.client import PostgresClient
from vehicletracker.azure.services import ServiceQueue

from datetime import datetime

import yaml

import json
import pandas as pd

_LOGGER = logging.getLogger('vehicletracker.trainer')

stop_event = threading.Event()
service_queue = ServiceQueue(domain = 'trainer', worker = True)
state_store = PostgresClient()  

# TODO: This will only work with a single trainer
class LocalJobRunner(object):

    def __init__(self):
        self.job_id = 0      
        self.jobs = [] 
        self.pending_jobs = queue.Queue()
        self.threads = []

    def worker(self):
        while True:
            item = self.pending_jobs.get()
            if item is None:
                break
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
        for i in range(1):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)

    def stop(self):
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

job_runner = LocalJobRunner()

def load_config():
    config_file = "configuration.yaml"
    with open(config_file, "r") as fd:
        config = yaml.load(fd.read())
    return config

def list_trainer_jobs(data):
    return job_runner.jobs

def schedule_train_link_model(data):
    link_ref = data['linkRef']
    model_name = data['model']
    _LOGGER.debug(f"Scheduling 'link model train' for link '{link_ref}' using model '{model_name}'.")
    return { 'jobId': job_runner.add_job(train, data)['jobId'] }
    
def train(data):
    import pandas as pd    
    
    link_ref = data['linkRef']    
    time = pd.to_datetime(data.get('time') or pd.datetime.now())
    model_name = data['model']
    model_parameters = data.get('parameters', {})
    _LOGGER.debug(f"Train link model for '{link_ref}' using model '{model_name}'.")   

    from vehicletracker.data.client import PostgresClient
    from vehicletracker.models import WeeklySvr

    import joblib

    data = PostgresClient()

    n = model_parameters.get('n', 21)
    train = data.link_travel_time_n_preceding_normal_days(link_ref, time, n)
    _LOGGER.debug(f"Loaded train data: {train.shape[0]}")

    weekly_svr = WeeklySvr()
    weekly_svr.fit(train.index, train.values)

    link_ref_safe = link_ref.replace(':', '-')
    MODEL_CACHE_PATH = './cache/lt-link-travel-time/'

    metadata_file_name = f'{model_name}-{time:%Y%m%d}-{n}-{link_ref_safe}.json'
    model_file_name = f'{model_name}-{time:%Y%m%d}-{n}-{link_ref_safe}.joblib'

    metadata = {
        'model': model_name,
        'linkRef': link_ref,
        'time': time.isoformat(),
    }

    # Write metadata
    with open(os.path.join(MODEL_CACHE_PATH, metadata_file_name), 'w') as f:
        json.dump(f, metadata)
    
    # Write model
    joblib.dump(weekly_svr, os.path.join(MODEL_CACHE_PATH, model_file_name))

    return metadata

def main():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    
    config = load_config()
    logging.config.dictConfig(config['logging'])

    service_queue.register_service('schedule_train_link_model', schedule_train_link_model)
    service_queue.register_service('list_trainer_jobs', list_trainer_jobs)
    service_queue.start()
    job_runner.start()

    input("Press Enter to exit...")
    os._exit(0)
    
    job_runner.stop()
    service_queue.stop()    

if __name__ == '__main__':
    main()
