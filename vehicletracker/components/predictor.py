"""Vehicle Tracker Predictor Component"""

import logging
from datetime import datetime

from typing import (
    Dict,
    Any
)

import pandas as pd

from vehicletracker.core import VehicleTrackerNode, callback
from vehicletracker.helpers.model_store import LocalModelStore

DOMAIN = 'predictor'

_LOGGER = logging.getLogger(__name__)

LINK_MODEL_PATH = './cache/lt-link-travel-time/'

LINK_MODELS = {}

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
        """Service handler for 'link_predict'. Predicts a single link at a single point in time."""

        link_ref = event['linkRef']
        time = datetime.fromisoformat(event.get('time')) if event.get('time') else datetime.now()
        model_name = event['model']
        model_candidates = self.link_model_store.list_models(model_name, link_ref, time)

        _LOGGER.debug("Recived link predict request for link '%s' using model '%s' at time %s.",
            link_ref, model_name, time)

        index = pd.DatetimeIndex([time])
        pred = model.predict(index)

        return pred[0, 0]

    def list_link_models(self, service_data):        
        """Service handler for 'list_link_models'. List available link models."""

        model_name = service_data.get('model')
        link_ref = service_data.get('linkRef')
        time = datetime.fromisoformat(service_data.get('time')) if service_data.get('time') else None
        return self.link_model_store.list_models(model_name, link_ref, time)

    @callback
    def link_model_available(self, event):
        """Event callback for 'link_model_available'"""
        self.link_model_store.add_model(event['metadata'])
