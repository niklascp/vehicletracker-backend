import logging

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Reshape

_LOGGER = logging.getLogger(__name__)

class DwellTimeModel:
    def __init__(self, node):
        self.node = node

    def dwell_time_from_to(self, from_time, to_time, line_ref = None, stop_point_ref = None):
        data = self.node.services.call('dwell_time_from_to', {
            'fromTime': from_time,
            'toTime': to_time,
            'lineRef': line_ref
        })
        if 'error' in data:
            raise ValueError(data['error'])
        return data.get('data'), data.get('labels')

class NnMultiStopDwell(DwellTimeModel):
    
    def __init__(self, node):
        self.node = node
    
    def build_model(self, params):
        self.stop_point_dim = params.get('stopPointDim', 4)
        self.day_hour_dim =  params.get('dayHourDim', 2)
        
        in_stop_point = Input(shape = (1, ), name = 'stop_point')
        in_day_time = Input(shape = (1, ), name = 'day_time')
        in_delay = Input(shape = (1, ), name = 'delay')

        l_stoppoint = Embedding(len(self.stop_point_ref), self.stop_point_dim)(in_stop_point)
        l_day_time = Embedding(4 * 24, self.day_hour_dim)(in_day_time)
        l_delay = Dense(1, name='fc_delay', activation='relu')(in_delay)
        l_delay = Reshape((1, 1), name='reshape_delay')(l_delay)
        l_out = Concatenate()([l_stoppoint, l_day_time, l_delay])
        out_dwell = Dense(1, name='fc', activation='relu')(l_out)
        model = Model(inputs = [in_stop_point, in_day_time, in_delay], outputs = [out_dwell])
        model.compile('rmsprop', 'mse')
        return model
    
    def dow_from_time(self, time):
        time = pd.to_datetime(time)
        hour = time.hour.values
        dow = np.zeros(len(time))
        dow[time.weekday == 4] = 1
        dow[time.weekday == 5] = 2
        dow[time.weekday == 6] = 3
        return (dow * 24 + hour).astype(int)
    
    def fit(self, data, labels, params):
        verbose = params.get('verbose', False)
        self.stop_point_ref = np.array(labels['stopPointRef'])
                
        time = pd.to_datetime(data['time'], unit='s')
        
        x_stop_point = np.array(data['stopPointRef']) 
        x_day_time = self.dow_from_time(time)
        x_delay =  np.array(data['delay'])
        y_dwell =  np.array(data['dwellTime'])
        
        self.keras_model = self.build_model(params)
        
        if verbose:
            print(self.keras_model.summary())
        
        self.history = self.keras_model.fit([x_stop_point, x_day_time, x_delay], [y_dwell], batch_size=100, epochs=50, validation_split = .2, verbose=verbose)    
    
    def train(self, params):        
        # Get dwell train data
        model_parameters = params['parameters'] or {}
        fromTime = str(pd.to_datetime(params['time']) - pd.DateOffset(days = model_parameters.get('nDays', 1)))
        data_train, labels_train = self.dwell_time_from_to(fromTime, params['time'], model_parameters.get('lineRef'), model_parameters.get('stopPointRef'))
        # Fit the model
        self.fit(data_train, labels_train, params)
        return {
            'type': 'dwell',
            'spatialRefs': self.stop_point_ref.tolist(),
            'trainSamples': len(data_train['time']),
            'history': self.history.history
        }
    
    def save(self, model_store, metadata):        
        self.keras_model.save(metadata['resourceUrl'])
    
    def restore(self, model_store, metadata):
        _LOGGER.info("Restoring model '%s'...", metadata['ref'])
        # model_store.download_to_cache()
        self.stop_point_ref = np.array(metadata['spatialRefs'])
        self.keras_model = keras.models.load_model(metadata['resourceUrl'])

    def predict(self, predict_params):
        # Single prediction
        if 'time' in predict_params and isinstance(predict_params['time'], str):
            x_stop_point = np.argwhere(self.stop_point_ref == predict_params['stopPointRef']).flatten()
            x_time = self.dow_from_time([predict_params['time']])
            x_delay = np.array([predict_params['delay']])
            return self.keras_model.predict([x_stop_point, x_time, x_delay]).flatten()
        # Batch prediction
        elif 'time' in predict_params and isinstance(predict_params['time'], list):
            x_stop_point = np.searchsorted(self.stop_point_ref, predict_params['stopPointRef'])
            x_time = self.dow_from_time(predict_params['time'])
            x_delay = np.array(predict_params['delay'])
            return self.keras_model.predict([x_stop_point, x_time, x_delay]).flatten()
        else:
            raise ValueError('Unsupported predict parameters: %s', predict_params)