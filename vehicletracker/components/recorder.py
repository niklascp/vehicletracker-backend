"""Vehicle Tracker Predictor Component"""

import logging
from typing import (
    Dict,
    List,
    Any
)
import json
from vehicletracker.core import VehicleTrackerNode, callback
from vehicletracker.helpers.json import DateTimeEncoder
from vehicletracker.const import EVENT_NODE_STOP, ATTR_NODE_NAME
from datetime import datetime

DOMAIN = 'recorder'

_LOGGER = logging.getLogger(__name__)

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Sets up the recorder component"""
    
    recorder = Recorder(node)

    # Wire up events and services
    _LOGGER.info("Starting record of events...")
    await node.events.async_listen('*', recorder.upon_event)

    return True

class Recorder():
    def __init__(self, node : VehicleTrackerNode) -> None:
        self.node = node
        self._json_encoder = DateTimeEncoder()
        self._file_handle = open('recorder.txt', 'w')
        self._file_handle.write("time;event_type;event_data\n")

    async def upon_event(self, event_type : str, event_data : Dict[str, Any]):
        if event_type in ['error_stat', 'estimated_arrival', 'estimated_departure', 'arrival', 'departure']:
            self._file_handle.write(f'{datetime.utcnow().isoformat()};{event_type};{self._json_encoder.encode(event_data)}\n')
        if event_type == EVENT_NODE_STOP and event_data [ATTR_NODE_NAME] == self.node.name:
            _LOGGER.info("Stopping record of events...")
            self._file_handle.close()
        