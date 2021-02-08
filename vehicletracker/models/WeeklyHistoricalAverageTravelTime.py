""" Link Travel Time Models for VehicleTracker """
import logging
from datetime import datetime
from typing import Any, Dict
import json

import numpy as np
import pandas as pd

from vehicletracker.helpers.spatial_reference import SpatialRef
from vehicletracker.models.travel_time import TravelTimeModel

_LOGGER = logging.getLogger(__name__)

class WeeklyHistoricalAverageTravelTime(TravelTimeModel):

    def train(self, time : datetime, spatial_ref : SpatialRef, parameters : Dict[str, Any]) -> Dict[str, Any]:
        n_days = parameters.get('nDays', 21)
        data, data_labels = self.travel_time_n_preceding_normal_days(time, n_days, spatial_ref)

        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        self.link_refs = np.array(data_labels['linkRef'])
        self.link_ix_lookup = { key: value for value, key in enumerate(self.link_refs) }
        self.travel_time_lookup = []
        
        for link_ix, link_ref in enumerate(self.link_refs):
            ix_hour = np.arange(24)
            ix_weekday = np.arange(7)
            df_link = df[df['linkRef'] == link_ix].drop('linkRef', axis=1)
            df_link['hour'] = df_link['time'].dt.hour
            df_link['weekday'] = df_link['time'].dt.weekday
            matrix = df_link.groupby(['hour', 'weekday'])['travelTime'].mean().unstack(1).reindex(ix_hour, axis=0).reindex(ix_weekday, axis=1)
            matrix = matrix.fillna(method='ffill').fillna(method='bfill').fillna(df_link['travelTime'].mean())
            self.travel_time_lookup.append(matrix)

        pred = self.predict({
            'time': df['time'].astype(str).tolist(),
            'linkRef': self.link_refs[df['linkRef']].tolist()
        })

        return {
            'loss': np.mean((df['travelTime'] - pred)**2),
            'spatialRefs': data_labels['linkRef']
        }

    def save(self, model_store, metadata):
        config = []
        for link_ix, link_ref in enumerate(self.link_refs):
            config.append({
                'linkRef': link_ref,
                'lookup': self.travel_time_lookup[link_ix].values.tolist()
            })

            with open(metadata['resourceUrl'] + '/config.json', 'w') as config_file:
                json.dump(config, config_file)
    
    def restore(self, model_store, metadata):
        _LOGGER.info("Restoring model '%s'...", metadata['ref'])

        with open(metadata['resourceUrl'] + '/config.json', 'r') as config_file:
            config = json.load(config_file)

        self.link_refs = []
        self.travel_time_lookup = []

        for link in config:
            ix_hour = np.arange(24)
            ix_weekday = np.arange(7)
            self.link_refs.append(link['linkRef'])
            self.travel_time_lookup.append(pd.DataFrame(data=link['lookup'], index=ix_hour, columns=ix_weekday))

        self.link_ix_lookup = { key: value for value, key in enumerate(self.link_refs) }

    def predict(self, predict_params):
        # Single prediction
        if 'time' in predict_params and isinstance(predict_params['time'], str):
            time = pd.to_datetime([predict_params['time']])
            link_ix = self.link_ix_lookup[predict_params['linkRef']]
            return self.travel_time_lookup[link_ix].lookup(time.hour, time.weekday)
        # Batch prediction
        elif 'time' in predict_params and isinstance(predict_params['time'], list):
            time = pd.to_datetime(predict_params['time'])
            link_ix = np.array([self.link_ix_lookup[x] for x in predict_params['linkRef']])
            pred = np.empty_like(time, dtype=float) 
            for link_ix_, link_ref_ in enumerate(self.link_refs):
                mask = link_ix_ == link_ix
                pred[mask] = self.travel_time_lookup[link_ix_].lookup(time[mask].hour, time[mask].weekday)
            return pred
        else:
            raise ValueError('Unsupported predict parameters: %s', predict_params)
