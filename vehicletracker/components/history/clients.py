"""Different clients for reading history data"""

import logging
from typing import (Any, Dict)

import numpy as np
import pandas as pd

from vehicletracker.core import VehicleTrackerNode

_LOGGER = logging.getLogger(__name__)

class HistoryDataSource():
    def dwell_time_from_to(self, params):
        pass

class MssqlHistoryDataSource(HistoryDataSource):
    """History data directly from MS SQL Data Warehouse"""

    def __init__(self):
        pass
    
    def async_setup(self, node : VehicleTrackerNode, config : Dict[str, Any]):
        from sqlalchemy import create_engine
        self.engine = create_engine(config['connection_string'])

    def calendar(self, params):
        """Get calendar information"""

        from_date = pd.to_datetime(params['fromDate'])
        to_date = pd.to_datetime(params['toDate'])

        data = pd.read_sql_query(
            'exec api.RT_VehicleTracker_Calendar @fromDate = ?, @toDate = ?',
            self.engine,
            params=[from_date, to_date])

        result = {
            'date': pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d').values.tolist(),
            'weekday': data['WeekDay'].values.tolist(),
            'day_type': data['DayType'].values.tolist(),
            'statutory_holiday': data['StatutoryHoliday'].values.tolist(),
        }

        _LOGGER.debug(f"returning {len(data)} results")

        return result

    def link_travel_time_from_to(self, params):
        """Get link travel times"""

        link_ref = params['linkRef']
        from_time = pd.to_datetime(params['fromTime'])
        to_time = pd.to_datetime(params['toTime'])

        data = pd.read_sql_query(
            'exec api.RT_VehicleTracker_LinkTavelTime @linkRef = ?, @fromTime = ?, @toTime = ?',
            self.engine,
            params=[link_ref, from_time, to_time])

        result = {
            'time': np.diff(np.hstack((0, data['time'].astype(np.int64) // 10**9))).tolist(),
            'link_travel_time': data['link_travel_time'].values.tolist()
        }

        _LOGGER.debug(f"returning {len(data)} results")

        return result

    def link_travel_time_n_preceding_normal_days(self, params):
        link_ref = params['linkRef']
        time = pd.to_datetime(params['time'])
        n = int(params['n'])

        data = pd.read_sql_query(
            'exec api.RT_VehicleTracker_LinkTavelTime_NPrecedingNormalDays @linkRef = ?, @time = ?, @n = ?',
            self.engine,
            params=[link_ref, time, n])

        result = {
            'time': np.diff(np.hstack((0, data['time'].astype(np.int64) // 10**9))).tolist(),
            'link_travel_time': data['link_travel_time'].values.tolist()
        }

        _LOGGER.debug(f"returning {len(data)} results")

        return result

    def link_travel_time_special_days(self, params):
        link_ref = params['linkRef']
        from_time = pd.to_datetime(params['fromTime'])
        to_time = pd.to_datetime(params['toTime'])

        data = pd.read_sql_query(
            'exec api.RT_VehicleTracker_LinkTavelTime_SpecialDays @fromTime = ?, @toTime = ?, @linkRef = ?',
            self.engine,
            params=[from_time, to_time, link_ref])

        result = {
            'time': np.diff(np.hstack((0, data['time'].astype(np.int64) // 10**9))).tolist(),
            'link_travel_time': data['link_travel_time'].values.tolist(),
            'day_type': data['day_type'].values.tolist()
        }

        _LOGGER.debug(f"returning {len(data)} results")

        return result

    def dwell_time_from_to(self, params):
        from_time = pd.to_datetime(params['fromTime'])
        to_time = pd.to_datetime(params['toTime'])

        sql = """
        select
            [time] = [p].[ObservedArrivalDateTime],
            [stop_point_ref] = [p].[StopPointNumber],
            [is_timing_point] = [p].[IsTimingPoint],
            [delay] = datediff(second, [p].[PlannedArrivalDateTime], [p].[ObservedArrivalDateTime]),
            [dwell_time] = datediff(second, [ObservedArrivalDateTime], [p].[ObservedDepartureDateTime])
        from
            [data].[RT_JourneyPoint] [p] (nolock)
        where
            [p].[OperatingDayDate] between dateadd(day, -1, '{0:}') and '{1:}'
            and [p].[ObservedArrivalDateTime] between '{0:}' and '{1:}'
            and [p].[ObservedArrivalDateTime] < [p].[ObservedDepartureDateTime]
            and [p].[IsStopPoint] = 1
        """.format(from_time, to_time)

        if params.get('lineRef'):
            sql += "    and [p].[LineNumber] = {:d}\n".format(int(params.get('lineRef')))
        elif params.get('stopPointRef'):
            sql += "    and [p].[StopPointNumber] = {:d}\n".format(int(params.get('stopPointRef')))
            if params.get('lineRef'):
                sql += "    and [p].[LineNumber] = {:d}\n".format(int(params.get('lineRef')))
        else:
            raise ValueError('Must provide lineRef and/or stopPointRef')

        data = pd.read_sql_query(sql, self.engine)
        stop_point_ref, stop_point_ref_labels = data['stop_point_ref'].astype(str).factorize(sort=True)
        epoch = pd.Timestamp("1970-01-01")

        result = {
            'labels': {
                'stopPointRef': stop_point_ref_labels.tolist()
            },
            'data': {
                'time': ((data['time'] - epoch) // pd.Timedelta('1s')).tolist(),
                'stopPointRef': stop_point_ref.tolist(),    
                'delay': data['delay'].tolist(),
                'isTimingPoint': data['is_timing_point'].tolist(),
                'dwellTime': data['dwell_time'].tolist(),
            } 
        }
        return result

class FileHistoryDataSource():
    """Local travel time history loaded from CSV-files"""

    def __init__(self):
        self.ready = False
        self.calendar = None
        self.link_travel_time = None

    def async_setup(self, node : VehicleTrackerNode, config : Dict[str, Any]):
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
