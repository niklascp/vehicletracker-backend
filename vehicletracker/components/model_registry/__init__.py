"""Vehicle Tracker Model Registry Component"""

import logging
from typing import (Any, Dict)

import importlib

from vehicletracker.core import callback, VehicleTrackerNode

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'model_registry'
ATTR_MODELS = 'models'
ATTR_NAME = 'name'
ATTR_CLASS = 'class'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup Model Regitry component"""

    component_config = config[DOMAIN]
    node.data[DOMAIN] = model_registry = ModelRegitry(component_config)

    await node.services.async_register(DOMAIN, 'list_models', model_registry.list_models)

    return True

class ModelRegitry():
    def __init__(self, config : Dict[str, Any]):
        self.models = {}
        for model_config in config[ATTR_MODELS]:
            # Initialie client
            model_module_name, model_class_name = model_config[ATTR_CLASS].rsplit(".", 1)
            model_class = getattr(importlib.import_module(model_module_name), model_class_name)
            self.models[model_config[ATTR_NAME]] = model_class

    def list_models(self, params):
        return list(self.models.keys())
