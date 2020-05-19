"""Vehicle Tracker Predictor Component"""

import logging
from datetime import datetime

from typing import (
    Dict,
    Any
)

import numpy as np
import pandas as pd

from vehicletracker.core import VehicleTrackerNode, callback
from vehicletracker.helpers.model_store import LocalModelStore
from vehicletracker.exceptions import ModelNotFound

DOMAIN = 'predictor'

_LOGGER = logging.getLogger(__name__)

LINK_MODEL_PATH = './cache/lt-link-travel-time/'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Sets up the predictor component"""

    predictor = node.data[DOMAIN] = Predictor(node, config[DOMAIN])
    
    node.async_add_job(predictor.restore_state)

    # Wire up events and services
    await node.services.async_register(DOMAIN, 'link_predict', predictor.predict)
    await node.services.async_register(DOMAIN, 'link_models', predictor.list_link_models)
    await node.events.async_listen('link_model_available', predictor.link_model_available)    

    return True

class Predictor():
    """Predictor State"""

    def __init__(self, node, config):
        self.link_model_store = LocalModelStore(LINK_MODEL_PATH)

    def restore_state(self):
        """Restore models from persistent model store"""
        _LOGGER.info('Loading cached models from persistent store ...')
        self.link_model_store.load_metadata()

    

    def predict(self, event):
        """Service handler for 'link_predict'. Predicts a single link at one or more points in time."""

        # link_ref and time are required, model is optional.
        link_ref = event['linkRef']
        time = event.get('time')
        model_name = event.get('model')
        
        if isinstance(time, str):
            time_min = datetime.fromisoformat(time)
            time_ = [time_min]
        elif isinstance(time, list):
            time_ = pd.to_datetime(np.cumsum(time), unit='s')
            time_min = time_.min()
        else:
            time_min = datetime.now()
            time_ = [time_min]

        model_candidates = self.link_model_store.list_models(model_name, link_ref, time_min)

        _LOGGER.debug("Recived link predict request for link '%s' using model '%s' at time %s.",
            link_ref, model_name, time_min)

        if len(model_candidates) == 0:
            _LOGGER.warning("No model condidates for link '%s' using model '%s' at time %s was found.", link_ref, model_name, time)
            return []

        model_ref = model_candidates[0]['ref']
        model = self.link_model_store.get_model(model_ref)

        index = pd.DatetimeIndex(time_)
        pred = model.predict(index)

        return [{
            'model': model_name,
            'model_ref': model_ref,
            'predicted': np.round(np.squeeze(pred), 1) if len(time_) == 1 else list(np.round(np.squeeze(pred), 1))
        }]

    def list_link_models(self, service_data):        
        """Service handler for 'list_link_models'. List available link models."""

        model_name = service_data.get('model')
        link_ref = service_data.get('linkRef')
        time = datetime.fromisoformat(service_data.get('time')) if service_data.get('time') else None
        return self.link_model_store.list_models(model_name, link_ref, time)

    @callback
    def link_model_available(self, event_type, event_data):
        """Event callback for 'link_model_available'"""
        self.link_model_store.add_model(event_data['metadata'])
