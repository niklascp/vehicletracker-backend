"""Vehicle Tracker Schedule Loader Component"""

import logging
from datetime import datetime
from typing import (Any, Dict)

import pandas as pd

from vehicletracker.core import callback, VehicleTrackerNode
from vehicletracker.helpers.events import async_track_utc_time_change

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'schedule_loader'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup schedule_loader component"""

    schedule_loader = node.data[DOMAIN] = DataWarehouseScheduleLoader(node, config[DOMAIN])

    await node.services.async_register(DOMAIN, 'load_stop_points', schedule_loader.load_stop_points)
    await node.services.async_register(DOMAIN, 'load_link_geometry', schedule_loader.load_link_geometry)

    await node.services.async_register(DOMAIN, 'load_journeys', schedule_loader.load_journeys)
    await node.services.async_register(DOMAIN, 'load_journey_stops', schedule_loader.load_journey_stops)    
    await node.services.async_register(DOMAIN, 'load_journey_links', schedule_loader.load_journey_links)

    return True

class DataWarehouseScheduleLoader():

    def __init__(self, node : VehicleTrackerNode, config : Dict[str, Any]):
        from sqlalchemy import create_engine
        self.engine = create_engine(config['connection_string'])

    def load_stop_points(self, service_data):
        """Service handler for 'load_stop_points'"""
        data = self.engine.execute(
            """
                select
                    [stopPointRef] = jpp.[StopPointId],
                    [name] = jpp.[StopPointName],
                    [latitude] = [JourneyPatternPointLatitude],
                    [longitude] = [JourneyPatternPointLongitude],
                    [arrivalRadius] = isnull(case
                        when sp.[TypeCode] = 'TRACK' then 30 + sp.[LengthMeters] / 2
                        when sp.[TypeCode] = 'BUSSTOP' then 30 + sp.[LengthMeters]
                    end, 30),
                    [departureRadius] =  isnull(case
                        when sp.[TypeCode] = 'TRACK' then 10 + sp.[LengthMeters] / 2
                        when sp.[TypeCode] = 'BUSSTOP' then 10 + sp.[LengthMeters]
                    end, 10)
                from
                    [dim].[JourneyPatternPoint] jpp
                    left join [data].[RT_DOI_StopPoint] sp on
                        sp.[StopPointId] = jpp.[StopPointId]
                        and jpp.ValidFromDate between sp.ValidFromDate and isnull(sp.ValidToDate, '9999-12-31')
                where
                    [IsCurrent] = 1
                    and jpp.JourneyPatternPointIsStopPoint = 1
            """).fetchall()

        return [dict(row) for row in data]


    def load_link_geometry(self, service_data):
        """Service handler for 'load_link_geometry'"""
        journey_ref = service_data['journeyRef']
        data = self.engine.execute(
            """
                with [JourneyLink] as
                (
                    select 
                        [LinkRef] = concat(lag(p.[StopPointNumber]) over (order by p.[SequenceNumber]), ':', p.[StopPointNumber])
                    from
                        [data].[RT_JourneyPoint] p
                    where
                        p.[JourneyRef] = ?
                        and p.IsStopPoint = 1                
                )
                select 
                    [linkRef] = rl.[LinkRef],
                    [geometryWkt] = [Geography].STAsText()     
                from
                    [data].[GIS_RouteLink_StopPoint] rl
                    join [JourneyLink] jl on jl.[linkRef] = rl.[LinkRef]
                where
                    [IsCurrent] = 1
            """, journey_ref).fetchall()

        return [dict(row) for row in data]

    def load_journeys(self, service_data):
        """Service handler for 'load_journeys'"""
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
    [data].[RT_Journey] j (nolock)
    join [dim].[JourneyPattern] jp on jp.[JourneyPatternId] = j.[JourneyPatternId] and jp.[IsCurrent] = 1
where
    --j.[LineNumber] in (1, 2, 4, 5, 6, 7, 9, 10, 15, 150, 375)
    j.[LineNumber] in (10, 15, 150, 375) and
    j.[JourneyNumber] < 1000
    and getdate() between dateadd(minute, -15, [PlannedStartDateTime]) and [PlannedEndDateTime]
order by
    [PlannedStartDateTime]
            """).fetchall()

        return [dict(row) for row in data]




    def load_journey_stops(self, service_data):
        """Service handler for 'load_journey_stops'. Load stops for a given journey."""
        
        journey_ref = service_data['journeyRef']

        _LOGGER.debug("Fetching stops for journey ref '%s'", journey_ref)

        data = self.engine.execute(
            """
select 
    [sequenceNumber] = [SequenceNumber],
    [stopPointRef] = cast(p.[StopPointNumber] as nvarchar(20)),
    [plannedArrivalUtc] = format(p.[PlannedArrivalDateTime] at time zone 'Central European Standard Time' at time zone 'UTC', 'yyyy-MM-ddTHH:mm:ssZ'),
    [plannedDepartureUtc] = format(p.[PlannedDepartureDateTime] at time zone 'Central European Standard Time' at time zone 'UTC', 'yyyy-MM-ddTHH:mm:ssZ')
from
    [data].[RT_JourneyPoint] p (nolock)
where
    p.[JourneyRef] = ?
    and p.IsStopPoint = 1
            """, journey_ref).fetchall()
        
        return [dict(row) for row in data]


    def load_journey_links(self, service_data):
        """Service handler for 'load_journey_links'"""
        
        journey_ref = service_data['journeyRef']

        _LOGGER.debug("Fetching links for journey ref '%s'", journey_ref)

        data = self.engine.execute(
            """
select
    *
from
(
    select 
        [sequenceNumber] = [SequenceNumber],
        [linkRef] = concat(lag(p.[StopPointNumber]) over (order by p.[SequenceNumber]), ':', p.[StopPointNumber]),
        [plannedTime] = datediff(second, lag(p.[PlannedDepartureDateTime]) over (order by p.[SequenceNumber]), p.[PlannedArrivalDateTime]),
        [totalDistance] = p.[PlannedJourneyDistanceMeters]
    from
        [data].[RT_JourneyPoint] p (nolock)
    where
        p.[JourneyRef] = ?
        and p.IsStopPoint = 1
) p
where
    p.[SequenceNumber] > 1
order by
    p.[SequenceNumber]
            """, journey_ref).fetchall()
        
        return [dict(row) for row in data]
