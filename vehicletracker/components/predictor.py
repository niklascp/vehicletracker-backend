import sys
import os
import logging

from vehicletracker.helpers.events import EventQueue
from vehicletracker.helpers.model_store import LocalModelStore

from datetime import datetime

import json

import pandas as pd

_LOGGER = logging.getLogger(__name__)

LINK_MODEL_PATH = './cache/lt-link-travel-time/'

LINK_MODELS = {}

event_queue = EventQueue(domain = 'predictor')

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
    link_model_store.add_model(event['metadata'])

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
