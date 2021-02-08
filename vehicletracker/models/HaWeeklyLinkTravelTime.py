""" Link Travel Time Models for VehicleTracker """
import json
import logging
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
from vehicletracker.helpers.spatial_reference import SpatialRef
from vehicletracker.models.travel_time import TravelTimeModel

_LOGGER = logging.getLogger(__name__)

class HaWeeklyLinkTravelTime(TravelTimeModel):

    def train(self, time : datetime, spatial_ref : SpatialRef, parameters : Dict[str, Any]) -> Dict[str, Any]:
        self.n_days = parameters.get('nDays', 21)
        self.freq = parameters.get('freq', '15min')

        index = pd.date_range('1970-01-01', '1970-01-08', closed='left', freq=self.freq)

        # TODO: Request history to drop file in blob and download blob to cache (or load as stream)
        data = pd.read_csv(f'./cache/weekly_average_link_travel_time_{self.freq}.csv.gz')
        data['time'] = pd.to_datetime('1970-01-01') + (data['dow'] - 1) * pd.to_timedelta('1D') + pd.to_timedelta(data['tod'])
        data_pivot = data.groupby(['time', 'link_ref'])['link_travel_time'].first().unstack(1)
        data_pivot = data_pivot.reindex(index, axis=0).interpolate(limit=3).fillna(method='ffill', limit=3).fillna(method='bfill', limit=3).fillna(data_pivot.mean())

        self.travel_time_lookup = data_pivot
        
        return { }

    def save(self, model_store, metadata):
        _LOGGER.info("Saving model... (hash: '%s')", metadata['hash'])
        config = {
            'nDays': self.n_days,
            'freq': self.freq 
        }
        with open(metadata['resourceUrl'] + '/config.json', 'w') as config_file:
            json.dump(config, config_file)
        self.travel_time_lookup.to_csv(metadata['resourceUrl'] + '/data.csv.gz', index=True, compression='gzip')
    
    def restore(self, model_store, metadata):
        _LOGGER.info("Restoring model '%s'...", metadata['ref'])
        with open(metadata['resourceUrl'] + '/config.json', 'r') as config_file:
            config = json.load(config_file)
        self.n_days = config['nDays']
        self.freq = config['freq']
        self.travel_time_lookup = pd.read_csv(metadata['resourceUrl'] + '/data.csv.gz', index_col=0, compression='gzip', parse_dates=True)

    def predict(self, predict_params):        
        # Single prediction
        if 'time' in predict_params and isinstance(predict_params['time'], str):
            if not predict_params['linkRef'] in self.travel_time_lookup.columns:
                return None
            time = pd.to_datetime(predict_params['time'])
            time_lookup = time.replace(year=1970, month=1, day=time.weekday() + 1, tzinfo=None).floor(self.freq)
            return self.travel_time_lookup.loc[time_lookup, predict_params['linkRef']]
        # Batch prediction
        elif 'time' in predict_params and isinstance(predict_params['time'], list):
            raise NotImplementedError('Batch predict not implemented')
        else:
            raise ValueError('Unsupported predict parameters: %s', predict_params)
