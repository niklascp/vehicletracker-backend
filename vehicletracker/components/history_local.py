"""Vehicle Tracker History Component"""

import logging
from datetime import datetime
from typing import (Any, Dict)

import numpy as np
import pandas as pd

from vehicletracker.core import callback, VehicleTrackerNode

_LOGGER = logging.getLogger(__name__)

DOMAIN = 'history'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Setup history component"""

    client = LocalTravelTimeHistory()
    node.add_job(client.preload)

    await node.services.async_register(DOMAIN, 'link_travel_time', client.link_travel_time_from_to)
    await node.services.async_register(DOMAIN, 'link_travel_time_n_preceding_normal_days', client.link_travel_time_n_preceding_normal_days)
    await node.services.async_register(DOMAIN, 'link_travel_time_special_days', client.link_travel_time_special_days)

    return True

class LocalTravelTimeHistory():
    """Local travel time history"""

    def __init__(self):
        self.ready = False
        self.calendar = None
        self.link_travel_time = None

    def preload(self):
        """Preloads data"""
        self.calendar = pd.read_csv('./data/calendar.csv', index_col = 0, parse_dates = True)
        self.link_travel_time = pd.read_csv('./data/link_travel_time_local.csv.gz', compression = 'gzip', index_col = 0, parse_dates = True)        
        _LOGGER.info(f'loaded local data: {len(self.link_travel_time)}')
        
        self.link_travel_time['day'] = self.calendar.loc[self.link_travel_time.index.date, 'day'].values
        self.link_travel_time['day_type'] = self.calendar.loc[self.link_travel_time.index.date, 'day_type'].values
        self.link_travel_time['date'] = pd.DatetimeIndex(self.link_travel_time.index.date)
        self.ready = True

    def link_travel_time_from_to(self, params):
        """Get link travel times"""

        if not self.ready:
            return

        link_ref = params['linkRef']
        from_time = pd.to_datetime(params['fromTime'])
        to_time = pd.to_datetime(params['toTime'])

        _LOGGER.debug(f"getting 'link_travel_time' for link '{link_ref}' between '{from_time}' and '{to_time}'")

        data = self.link_travel_time[from_time:to_time - pd.to_timedelta('1ns')][lambda x: x['link_ref'] == link_ref]
        result = {
            'time': np.diff(np.hstack((0, data.index.astype(np.int64) // 10**9))).tolist(),
            'link_travel_time': data['link_travel_time'].values.tolist(),
            #'day_type': data['day_type'].values.tolist()
        }
        
        _LOGGER.debug(f"returning {len(data)} results")

        return result

    def link_travel_time_n_preceding_normal_days(self, params):
        if not self.ready:
            return
        link_ref = params['linkRef']
        time = pd.to_datetime(params['time'])
        n = int(params['n'])

        _LOGGER.debug(f"getting 'link_travel_time_n_preceding_normal_days' for link '{link_ref}' at time '{time}' using n = {n}")
        dates = self.calendar[:time - pd.to_timedelta('1ns')][lambda x: x['day_type'] == x['day']][-n:].index
        data = self.link_travel_time[lambda x: (x['link_ref'] == link_ref) & (x['date'].isin(dates))]

        result = {
            'time': np.diff(np.hstack((0, data.index.astype(np.int64) // 10**9))).tolist(),
            'link_travel_time': data['link_travel_time'].values.tolist()
        }

        _LOGGER.debug(f"returning {len(data)} results")
        
        return result

    def link_travel_time_special_days(self, params):
        if not self.ready:
            return
        link_ref = params['linkRef']
        from_time = pd.to_datetime(params['fromTime'])
        to_time = pd.to_datetime(params['toTime'])

        _LOGGER.debug(f"getting 'link_travel_time_special_days' for link '{link_ref}' between '{from_time}' and '{to_time}'")

        data = self.link_travel_time[from_time:to_time - pd.to_timedelta('1ns')][lambda x: (x['link_ref'] == link_ref) & (x['day_type'] != x['day'])]
        result = {
            'time': np.diff(np.hstack((0, data.index.astype(np.int64) // 10**9))).tolist(),
            'link_travel_time': data['link_travel_time'].values.tolist(),
            'day_type': data['day_type'].values.tolist()
        }
        
        _LOGGER.debug(f"returning {len(data)} results")

        return result

