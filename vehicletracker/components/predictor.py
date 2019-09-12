import sys
import os
import logging

from vehicletracker.helpers.events import EventQueue

from datetime import datetime

import json

import pandas as pd

_LOGGER = logging.getLogger(__name__)

LINK_MODEL_PATH = './cache/lt-link-travel-time/'

LINK_MODELS = {}

event_queue = EventQueue(domain = 'predictor')

class LocalModelStore():
    import joblib

    def __init__(self, path):
        self.path = path

    def load_metadata(self):
        self.models = []

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for file_name in os.listdir(self.path):
            if not file_name.endswith(".json"):
                continue

            metadata_file_path = os.path.join(self.path, file_name)

            with open(metadata_file_path, 'r') as f:
                model_metadata = json.load(f)

            self.models.append(model_metadata)

    def list_models(self):
        return self.models

    def load_model(self, model_name):
        pass
        #for file_name in os.listdir(MODEL_CACHE_PATH):
        #    if (file_name.endswith(".json")):
        #        _LOGGER.info(f'Loading cached model from data from {file_name}')
        #        metadata_file_path = os.path.join(MODEL_CACHE_PATH, file_name)
        #        model_file_path = os.path.splitext(os.path.join(MODEL_CACHE_PATH, file_name))[0] + '.joblib'
        #        
        #        with open(metadata_file_path, 'r') as f:
        #            model_metadata = json.load(f)
        #        with open(model_file_path, 'rb') as f:
        #            model = joblib.load(f)
        #        
        #        LINK_MODELS[model_metadata['linkRef']] = { 'model': model, 'metadata': model_metadata }

link_model_store = LocalModelStore(LINK_MODEL_PATH)

def predict(event):
    link_ref = event['linkRef']
    time = datetime.fromisoformat(event.get('time')) if event.get('time') else datetime.now()
    model_name = event['model']
    model = LINK_MODELS[link_ref]['model']

    _LOGGER.debug(f"Recived link predict request for link '{link_ref}' using model '{model_name}' at time {time}.")

    ix = pd.DatetimeIndex([time])
    pred = model.predict(ix)

    return pred[0, 0]

def list_link_models(service_data):
    return link_model_store.list_models()

def link_model_available(event):
    #link_model_store.refresh(event[''])
    pass

def start(): 
    # TODO: Move into class DiskModelCache

    _LOGGER.info('Loading cached models from persistent store ...')
    link_model_store.load_metadata()

    # new
    event_queue.register_service('link_predict', predict)
    event_queue.register_service('link_models', list_link_models)
    event_queue.listen('link_model_available', link_model_available) #TODO: listen_all
    
    event_queue.start()

def stop():
    event_queue.stop()
