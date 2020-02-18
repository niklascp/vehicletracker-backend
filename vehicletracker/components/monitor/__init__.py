"""Vehicle Tracker History Component"""

import logging
from datetime import datetime
from typing import (Any, Dict)

from vehicletracker.core import callback, VehicleTrackerNode
from vehicletracker.helpers.events import async_track_utc_time_change

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'monitor'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup monitor component"""

    monitor = node.data[DOMAIN] = Monitor()

    await node.events.async_listen('linkCompleted', monitor.link_completed)
    await async_track_utc_time_change(node, monitor.fetch_journeys, hour='*', minute='*', second=0)

    return True

class Monitor():

    def link_completed(self, event):
        """Event handler for 'linkCompleted'. Collects true link time for error calculation."""

        link_ref = event['linkRef']
        journey_ref = event['journeyRef']
        sequence_number = event['sequenceNumber']
        
    def fetch_journeys(self, utc_time):
        _LOGGER.info('fetch_journeys %s', utc_time)
