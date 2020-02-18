"""Vehicle Tracker History Component"""

import logging
from datetime import datetime
from typing import (Any, Dict)

import pandas as pd

from vehicletracker.core import callback, VehicleTrackerNode
from vehicletracker.helpers.events import async_track_utc_time_change

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'monitor'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup monitor component"""

    monitor = node.data[DOMAIN] = Monitor(node, config[DOMAIN])

    await node.events.async_listen('linkCompleted', monitor.link_completed)
    await node.services.async_register(DOMAIN, 'journeys', monitor.list_journeys)
    await async_track_utc_time_change(node, monitor.fetch_journeys, hour='*', minute='*', second=0)

    return True

class Monitor():

    def __init__(self, node : VehicleTrackerNode, config : Dict[str, Any]):
        from sqlalchemy import create_engine
        self.engine = create_engine(config['connection_string'])
        self.journey_map = {}

    def link_completed(self, event):
        """Event handler for 'linkCompleted'. Collects true link time for error calculation."""

        link_ref = event['linkRef']
        journey_ref = event['journeyRef']
        sequence_number = event['sequenceNumber']
        
    def list_journeys(self, event_data):
        return self.journey_map

    def fetch_journeys(self, utc_time):
        # TODO: Journey Fetch should be splitted to own journey service, such that the moniter is not requireing direct access to data.
        _LOGGER.info('fetch_journeys %s', utc_time)
        results = []
        data = self.engine.execute(
            """
select 
    [journeyRef] = j.[JourneyRef],
    [lineDesignation] = j.[LineDesignation],
    [plannedStartDateTime] = j.[PlannedStartDateTime],
    [plannedEndDateTime] = j.[PlannedEndDateTime],
    [origin] = jp.[JourneyPatternStartStopPointName],
    [destination] = jp.[JourneyPatternEndStopPointName]
from
    [data].[RT_Journey] j
    join [dim].[JourneyPattern] jp on jp.[JourneyPatternId] = j.[JourneyPatternId] and jp.[IsCurrent] = 1
where
    j.[LineNumber] = 1
    and getdate() between dateadd(minute, -45, [PlannedStartDateTime]) and [PlannedEndDateTime]
order by
    [PlannedStartDateTime]
            """).fetchall()
        
        if len(data)==0:
            return

        for row in data:
            if not row['journeyRef'] in self.journey_map:
                self.journey_map[row['journeyRef']] = {column: value for column, value in row.items()}
                self.journey_map[row['journeyRef']]['added'] = utc_time

        print(self.journey_map)