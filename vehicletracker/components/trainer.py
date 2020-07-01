import sys
import os
import logging
import logging.config

from typing import (Any, Dict)

from vehicletracker.exceptions import ApplicationError
from vehicletracker.core import VehicleTrackerNode
from vehicletracker.helpers.job_runner import LocalJobRunner

from datetime import datetime

import yaml

import json
import pandas as pd

DOMAIN = 'trainer'
_LOGGER = logging.getLogger(__name__)
MODEL_CACHE_PATH = './cache/lt-link-travel-time/'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup trainer component"""
    node.data['trainer'] = trainer = Trainer(node)

    await node.services.async_register(DOMAIN, 'link_model_schedule_train', trainer.schedule_train_link_model)
    await node.services.async_register(DOMAIN, 'list_trainer_jobs', trainer.list_trainer_jobs)    
    # Old service naming:
    await node.services.async_register(DOMAIN, 'link_train', trainer.schedule_train_link_model)
    await node.services.async_register(DOMAIN, 'trainer_jobs', trainer.list_trainer_jobs)    

    return True

class Trainer():
    """Represent the trainer state"""

    def __init__(self, node):
        """Initializes the trainer state"""
        self.node = node
        self.trainer_job_count = 1
        self.trainer_job_task = {}
        self.trainer_job_state = {}

    def list_trainer_jobs(self, data):
        """List trainer jobs"""
        return sorted(self.trainer_job_state.values(), key = lambda x: x['jobId'])

    def schedule_train_link_model(self, params):
        """Schedule a new training of a model."""
        link_ref = params['linkRef']
        model_name = params['model']
        _LOGGER.info("Scheduling 'link model train' for link '%s' using model '%s'.", link_ref, model_name)
        
        job_id = self.trainer_job_count 
        self.trainer_job_count += 1

        self.trainer_job_state[job_id] = job_state = {
            'jobId': job_id,
            'status': 'new',
            'input': params
        }

        def execute_train_job():
            try:
                job_state['status'] = 'running'
                job_state['started'] = datetime.now().isoformat()
                job_state['result'] = self.train(job_state['input'])
                job_state['status'] = 'completed'
                job_state['stopped'] = datetime.now().isoformat()
            except Exception as e: # pylint: disable=broad-except
                job_state['status'] = 'failed'
                job_state['stopped'] = datetime.now().isoformat()
                job_state['error'] = str(e)
                _LOGGER.exception('Error in execute_train_job')

        self.trainer_job_task[job_id] = self.node.add_job(execute_train_job)
        
        return { 'jobId': job_id }

    

    def train(self, params):
        """Performs the actual execution of training"""
        link_ref = params['linkRef']
        time = pd.to_datetime(params.get('time') or pd.datetime.now())
        model_name = params['model']
        model_parameters = params.get('parameters', {})

        import hashlib

        model_hash = hashlib.sha256(json.dumps({
            'linkRef': link_ref,
            'modelName': model_name,
            'time': time.isoformat(),
            'model_parameters': model_parameters
            }, sort_keys=True).encode('utf-8')).digest()
        model_hash_hex = ''.join('{:02x}'.format(x) for x in model_hash)

        _LOGGER.debug("Train link model for '%s' using model '%s' (hash: %s)", link_ref, model_name, model_hash_hex)

        from vehicletracker.models import WeeklySvr
        from vehicletracker.models import WeeklyHistoricalAverage

        import joblib

        n = model_parameters.get('n', 21)
        train_data = self.node.services.call('link_travel_time_n_preceding_normal_days', {
            'linkRef': link_ref,
            'time': time.isoformat(),
            'n': n
        }, timeout = 10)

        if 'error' in train_data:
            raise ApplicationError(f"error getting train data: {train_data['error']}")

        if len(train_data['time']) == 0:
            raise ApplicationError(f"no train data returned for '{link_ref}' (time: {time.isoformat()}, {n})")

        train = pd.DataFrame(train_data)
        train.index = pd.to_datetime(train['time'].cumsum(), unit='s')
        train.drop(columns = 'time', inplace=True)
        _LOGGER.debug("Loaded train data: %s", train.shape[0])

        metadata_file_name = f'{model_hash_hex}.json'
        model_file_name = f'{model_hash_hex}.joblib'

        if model_name == 'svr':
            weekly_svr = WeeklySvr(verbose = False)
            weekly_svr.fit(train.index, train.values)
            # Write model
            joblib.dump(weekly_svr, os.path.join(MODEL_CACHE_PATH, model_file_name))
        elif model_name == 'ha':
            weekly_ha = WeeklyHistoricalAverage()
            weekly_ha.fit(train.index, train.values)
            # Write model
            joblib.dump(weekly_ha, os.path.join(MODEL_CACHE_PATH, model_file_name))

        metadata = {
            'hash': model_hash_hex,
            'model': model_name,
            'linkRef': link_ref,
            'time': time.isoformat(),
            'trained': datetime.now().isoformat(),
            'resourceUrl': os.path.join(MODEL_CACHE_PATH, model_file_name)
        }

        # Write metadata
        with open(os.path.join(MODEL_CACHE_PATH, metadata_file_name), 'w') as f:
            json.dump(metadata, f)
        
        self.node.events.publish('link_model_available', {
            'metadata': metadata
        })

        return metadata
