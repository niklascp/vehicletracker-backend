import sys
import os
import logging
import logging.config

from vehicletracker.exceptions import ApplicationError
from vehicletracker.helpers.events import EventQueue
from vehicletracker.helpers.job_runner import LocalJobRunner

from datetime import datetime

import yaml

import json
import pandas as pd

_LOGGER = logging.getLogger(__name__)

job_runner = LocalJobRunner()
service_queue = EventQueue(domain = 'trainer')

def list_trainer_jobs(data):
    return job_runner.jobs

def schedule_train_link_model(data):
    link_ref = data['linkRef']
    model_name = data['model']
    _LOGGER.debug(f"Scheduling 'link model train' for link '{link_ref}' using model '{model_name}'.")
    return { 'jobId': job_runner.add_job(train, data)['jobId'] }
    
def train(params):
    import pandas as pd    
    
    link_ref = params['linkRef']    
    time = pd.to_datetime(params.get('time') or pd.datetime.now())
    model_name = params['model']
    model_parameters = params.get('parameters', {})
    _LOGGER.debug(f"Train link model for '{link_ref}' using model '{model_name}'.")   

    from vehicletracker.models import WeeklySvr

    import joblib

    n = model_parameters.get('n', 21)
    train_data = service_queue.call_service('link_travel_time_n_preceding_normal_days', {
        'linkRef': link_ref,
        'time': time.isoformat(),
        'n': n
    }, timeout = 5)

    if 'error' in train_data:
        raise ApplicationError(f"error getting train data: {train_data['error']}")

    if len(train_data['time']) == 0:
        raise ApplicationError(f"no train data returned for '{link_ref}' (time: {time.isoformat()}, {n})")

    train = pd.DataFrame(train_data)
    train.index = pd.to_datetime(train['time'].cumsum(), unit='s')
    train.drop(columns = 'time', inplace=True)
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
        'trained': datetime.now().isoformat()
    }

    # Write metadata
    with open(os.path.join(MODEL_CACHE_PATH, metadata_file_name), 'w') as f:
        json.dump(metadata, f)
    
    # Write model
    joblib.dump(weekly_svr, os.path.join(MODEL_CACHE_PATH, model_file_name))

    service_queue.publish_event({
        'eventType': 'link_model_available',
        'metadata': metadata
    })

    return metadata

def start(): 
    service_queue.register_service('link_model_schedule_train', schedule_train_link_model)
    service_queue.register_service('list_trainer_jobs', list_trainer_jobs)
    service_queue.start()
    job_runner.start()
    
def stop():
    service_queue.stop()
    job_runner.stop()
