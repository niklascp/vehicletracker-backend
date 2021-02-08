import sys
import os
import shutil
import logging

from typing import (Any, Dict, Type)

from vehicletracker.exceptions import ApplicationError
from vehicletracker.core import VehicleTrackerNode
from vehicletracker.helpers.job_runner import LocalJobRunner
from vehicletracker.helpers.spatial_reference import SpatialRef, parse_spatial_ref
from vehicletracker.components.model_registry import ModelRegitry
from vehicletracker.components.model_registry import DOMAIN as MODEL_REGISTRY

from datetime import datetime

import yaml
import json

import hashlib
import pandas as pd

DOMAIN = 'trainer'
_LOGGER = logging.getLogger(__name__)

MODEL_PATH = './cache/models/'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup trainer component"""

    model_registry : ModelRegitry = node.data[MODEL_REGISTRY]
    trainer = node.data[DOMAIN] = Trainer(node, model_registry)

    await node.services.async_register(DOMAIN, 'schedule_train_model', trainer.schedule_train_model)
    await node.services.async_register(DOMAIN, 'list_trainer_jobs', trainer.list_trainer_jobs)    

    #TODO:
    await node.services.async_register(DOMAIN, 'trainer_jobs', trainer.list_trainer_jobs)    

    return True

class Trainer():
    """Represent the trainer state"""

    def __init__(self, node : VehicleTrackerNode, model_registry : ModelRegitry):
        """Initializes the trainer state"""
        self.node = node
        self.trainer_job_count = 1
        self.trainer_job_task = {}
        self.trainer_job_state = {}
        self.model_registry = model_registry

    def list_trainer_jobs(self, data):
        """List trainer jobs"""
        return sorted(self.trainer_job_state.values(), key = lambda x: x['jobId'])

    def schedule_train_model(self, params):
        """Schedule a new training of a model."""
        model_name = params['model']
        if params['time'] == 'latest':
            time = datetime.now()
            time_txt = 'latest'
        else:
            time = datetime.fromisoformat(params['time'])
            time_txt = time.isoformat()
        spatial_ref = parse_spatial_ref(params['spatialRef'])
        model_parameters = params.get('parameters', {})

        model_class = self.model_registry.model_classes.get(model_name) # type: Type
        model_hash = hashlib.sha256(json.dumps({
            'model': model_name,
            'time': time_txt,
            'spatialRef': str(spatial_ref),
            'parameters': model_parameters
            }, sort_keys=True).encode('utf-8')).digest()
        model_hash_hex = ''.join('{:02x}'.format(x) for x in model_hash)
        
        _LOGGER.info("Scheduling model train for '%s' (hash: %s).", model_name, model_hash_hex)

        job_id = self.trainer_job_count 
        self.trainer_job_count += 1

        self.trainer_job_state[job_id] = job_state = {
            'jobId': job_id,
            'status': 'new',
            'input': params
        }

        def execute_train_job():
            _LOGGER.debug("Starting model train for '%s' (hash: %s).", model_name, model_hash_hex)
            try:
                model = model_class(self.node) # type: BaseModel
                
                job_state['status'] = 'running'
                job_state['started'] = datetime.now().isoformat()
                job_state['result'] = result = model.train(time, spatial_ref, model_parameters)
                job_state['status'] = 'saving'

                model_local_path = os.path.join(MODEL_PATH, model_name, model_hash_hex)
                if os.path.exists(model_local_path) and os.path.isdir(model_local_path):
                    shutil.rmtree(model_local_path)
                os.makedirs(model_local_path)                

                metadata = {
                    'hash': model_hash_hex,
                    'model': model_name,
                    'type':  model.model_type(),
                    'spatialRef': str(spatial_ref),
                    'time': time_txt,
                    'trained': datetime.now().isoformat(),
                    'parameters': model_parameters,
                    'resourceUrl': model_local_path
                }

                # Legacy for models that specify a list of spatial references
                if 'spatialRefs' in result:
                    metadata['spatialRefs'] = result['spatialRefs']
                    
                model.save(self.model_registry.model_store, metadata)

                # Write metadata
                with open(os.path.join(model_local_path, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)

                job_state['status'] = 'completed'
                job_state['stopped'] = datetime.now().isoformat()

                # Broadcast availability to all nodes
                self.node.events.publish('model_available', {
                    'metadata': metadata
                })

            except Exception as e: # pylint: disable=broad-except
                job_state['status'] = 'failed'
                job_state['stopped'] = datetime.now().isoformat()
                job_state['error'] = str(e)
                _LOGGER.exception('Error in execute_train_job')

        self.trainer_job_task[job_id] = self.node.add_job(execute_train_job)
        
        return { 'jobId': job_id }
