""" Link Travel Time Models for VehicleTracker """
import logging
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Concatenate, Dense, Dropout, Embedding,
                                     Input, Reshape)

from vehicletracker.helpers.spatial_reference import SpatialRef
from vehicletracker.models.travel_time import TravelTimeModel

_LOGGER = logging.getLogger(__name__)

class NnMultiLinkTravel(TravelTimeModel):

    def loss_fcn(self, y_true, y_pred, w):
        loss = tf.keras.losses.mse(y_true*w, y_pred*w)
        return loss
            
    def build_model(self, n_links, params):
        self.day_hour_dim = params.get('dayHourDim', 4)
        self.lr = params.get('lr', 0.01)
        self.dropout_prop = params.get('dropoutProp', 0.1)
        in_day_time = Input(shape = (1, ), name = 'day_time')
        in_y_true = Input(shape = (n_links, ), name = 'y_true')
        in_mask = Input(shape = (n_links, ), name = 'mask')
        

        l_day_time = Embedding(7 * 24, self.day_hour_dim, name='embed_day_time')(in_day_time)
        l_day_time = Reshape((self.day_hour_dim,), name='reshape_day_time')(l_day_time)
        out_run = Dense(n_links // 3, name='fc_1', activation='relu')(l_day_time)
        out_run = Dropout(self.dropout_prop, name='dropout')(out_run)
        out_run = Dense(n_links, name='fc_out', activation='relu')(out_run)

        opt = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        model = Model(inputs = [in_day_time, in_y_true, in_mask], outputs = [out_run])
        model.add_loss(self.loss_fcn(in_y_true, out_run, in_mask))
        model.compile(opt)
        return model
    
    def remove_outlier(self, data_frame):
        for grp, data in data_frame.groupby('linkRef'):
            x = data['travelTime']
            mad = 1.4826 * np.median(np.abs(x - np.median(x)))
            mask = np.abs(x - np.median(x)) / mad < 5
            data_frame.loc[data.index, 'mask'] = mask
        return data_frame[lambda x: (x['travelTime'] > 0) & x['mask']].drop('mask', axis=1).copy()
    
    def dow_from_time(self, time):
        time = pd.to_datetime(time)
        hour = time.hour.values
        dow = np.zeros(len(time))
        return (time.weekday * 24 + hour).astype(int).values
    
    def fit(self, data, labels, params):
        """ The actual training of the ANN. """
        verbose = params.get('verbose', False)
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        n_pre_outlier = len(df)
        df = self.remove_outlier(df)
        n_post_outlier = len(df)
        n_outlier_pct = (n_pre_outlier - n_post_outlier) / n_pre_outlier
        if verbose:
            print(f'Removed {n_outlier_pct} outliers.')
        
        df_train_resample = df.set_index('time').groupby('linkRef')['travelTime'].resample('1H').mean().unstack(0).dropna(how='all')
        x_train_time = self.dow_from_time(df_train_resample.index)
        Y_train = df_train_resample[df_train_resample.columns[df_train_resample.isnull().mean() < .3]]
        Y_train_filled = Y_train.fillna(method='ffill', limit=3).fillna(method='bfill', limit=3).fillna(Y_train.mean()).values
        X_mask = Y_train.notnull().astype(float).values
        
        self.link_refs = np.array(labels['linkRef'])[Y_train.columns]
        self.link_ix_lookup = { key: value for value, key in enumerate(self.link_refs) }
        self.keras_model = self.build_model(len(self.link_refs), params)
        
        if verbose:
            print(self.keras_model.summary())
        
        fit_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ]
        self.history = self.keras_model.fit(
            [x_train_time, Y_train_filled, X_mask], 
            Y_train_filled, 
            epochs=500, 
            batch_size=100,
            validation_split = .2, 
            callbacks=fit_callbacks,
            verbose=verbose)
    
    def train(self, time : datetime, spatial_ref : SpatialRef, parameters : Dict[str, Any]) -> Dict[str, Any]:
        """ Implements the train method, which also loads data. """
        verbose = parameters.get('verbose', False)
        n_days = parameters.get('nDays', 21)
        data, data_labels = self.travel_time_n_preceding_normal_days(time, n_days, spatial_ref)

        # Fit the model
        self.fit(data, data_labels, parameters)
        return {
            'spatialRefs': self.link_refs.tolist(),
            'trainSamples': len(data['time']),
            'history': self.history.history
        }

    def save(self, model_store, metadata):
        self.keras_model.save(metadata['resourceUrl'])
    
    def restore(self, model_store, metadata):
        _LOGGER.info("Restoring model '%s'...", metadata['ref'])
        # model_store.download_to_cache()
        self.link_refs = np.array(metadata['spatialRefs'])
        # Todo use semaphore - kares unly support one model loading at a time! This will be called from the ThreadPool!
        self.keras_model = tf.keras.models.load_model(metadata['resourceUrl'])
        self.link_ix_lookup = { key: value for value, key in enumerate(self.link_refs) }

    def predict(self, predict_params):
        # Single prediction
        if 'time' in predict_params and isinstance(predict_params['time'], str):
            link_ref_ix = self.link_ix_lookup[predict_params['linkRef']]
            x_time = self.dow_from_time([predict_params['time']])
            return self.keras_model([x_time]).numpy()[:, link_ref_ix].round(1)
        # Batch prediction
        elif 'time' in predict_params and isinstance(predict_params['time'], list):
            x_time = self.dow_from_time(predict_params['time'])
            y_pred = self.keras_model([x_time]).numpy()
            link_ref_ixs = np.array(list(map(lambda x: self.link_ix_lookup[x] if x in  self.link_ix_lookup else -1, predict_params['linkRef'])))
            return [y_pred[i, link_ref_ixs[i]].round(1) if link_ref_ixs[i] >= 0 else 0 for i in range(len(x_time))]
        else:
            raise ValueError('Unsupported predict parameters: %s', predict_params)
