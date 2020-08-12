"""Vehicle Tracker History Component"""

import logging
from datetime import datetime
from typing import (Any, Dict)

import numpy as np
import pandas as pd

import importlib

from vehicletracker.core import callback, VehicleTrackerNode

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'history'
ATTR_DATA_SOURCE = 'data_source'
ATTR_CLASS = 'class'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup history component"""

    component_config = config[DOMAIN]
    client_config = config[DOMAIN][ATTR_DATA_SOURCE]

    # Initialie client
    client_module_name, client_class_name = client_config[ATTR_CLASS].rsplit(".", 1)
    client_class = getattr(importlib.import_module(client_module_name), client_class_name)
    client = client_class()
    
    if hasattr(client, 'async_setup'):
        await node.async_add_job(client.async_setup, node, client_config)

    await node.services.async_register(DOMAIN, 'calendar', client.calendar)  
    await node.services.async_register(DOMAIN, 'link_travel_time', client.link_travel_time_from_to)
    await node.services.async_register(DOMAIN, 'link_travel_time_n_preceding_normal_days', client.link_travel_time_n_preceding_normal_days)
    await node.services.async_register(DOMAIN, 'link_travel_time_special_days', client.link_travel_time_special_days)

    await node.services.async_register(DOMAIN, 'dwell_time_from_to', client.dwell_time_from_to)

    return True
