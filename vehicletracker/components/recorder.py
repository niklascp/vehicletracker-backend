import logging

import json

from vehicletracker.helpers.events import EventQueue

_LOGGER = logging.getLogger(__name__)
_RECORDER_LOGGER = logging.getLogger('recorder')

event_queue = EventQueue(domain = 'recorder')

def log_event(event):
    _RECORDER_LOGGER.info(json.dumps(event))

def start(): 
    event_queue.listen('*', log_event)
    event_queue.start()

def stop(): 
    pass