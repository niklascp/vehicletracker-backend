"""Vehicle Tracker Model Registry Component"""

from datetime import datetime
import logging
from typing import (Any, Dict)

import importlib

from vehicletracker.core import callback, VehicleTrackerNode
from vehicletracker.helpers.model_store import LocalModelStore

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'model_registry'
ATTR_MODELS = 'models'
ATTR_NAME = 'name'
ATTR_CLASS = 'class'

METADATA_ATTR_MODEL_REF = 'ref'
METADATA_ATTR_MODEL_NAME = 'model'

MODEL_CACHE_PATH = './cache/models/'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup Model Regitry component"""

    component_config = config[DOMAIN]
    node.data[DOMAIN] = model_registry = ModelRegitry(node, component_config)

    node.async_add_job(model_registry.restore_state)
    await node.services.async_register(DOMAIN, 'list_model_classes', model_registry.list_model_classes)
    await node.services.async_register(DOMAIN, 'list_models', model_registry.list_models)
    await node.services.async_register(DOMAIN, 'link_models', model_registry.list_models) #TODO: Rename in frontend

    await node.events.async_listen('model_available', model_registry.model_available)

    return True

class ModelRegitry():
    def __init__(self, node : VehicleTrackerNode, config : Dict[str, Any]):
        self.node = node
        self.model_store = LocalModelStore(MODEL_CACHE_PATH)
        self.model_classes = {}
        for model_config in config[ATTR_MODELS]:
            # Initialie client
            model_module_name, model_class_name = model_config[ATTR_CLASS].rsplit(".", 1)
            model_class = getattr(importlib.import_module(model_module_name), model_class_name)
            self.model_classes[model_config[ATTR_NAME]] = model_class
        
    def restore_state(self):
        """Restore models from persistent model store"""
        _LOGGER.info('Loading cached models from persistent store ...')
        self.model_store.load_metadata()

    def get_model(self, model_metadata):
        model_name = model_metadata[METADATA_ATTR_MODEL_NAME]

        # Initialize a new instance from model_store
        model = self.model_classes[model_name](self.node)
        model.restore(self.model_store, model_metadata)
        return model

    def list_model_classes(self, service_data):
        """Service handler for 'list_models'. List available model classes."""
        return list(self.model_classes.keys())

    def list_models(self, service_data):        
        """Service handler for 'list_models'. List available models."""
        model_name = service_data.get('model')
        spatial_ref = service_data.get('spatialRef')
        time = datetime.fromisoformat(service_data.get('time')) if service_data.get('time') else None
        return self.model_store.list_models(model_name, spatial_ref, time)

    def model_available(self, event_name, event_data): 
        self.model_store.add_model(event_data['metadata'])
