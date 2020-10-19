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

from vehicletracker.core import VehicleTrackerNode
from vehicletracker.models.travel_time import TravelTimeModel
from vehicletracker.helpers.spatial_reference import SpatialRef
from vehicletracker.models.travel_time import TravelTimeModel

_LOGGER = logging.getLogger(__name__)

class RnnMultiLinkTravel(TravelTimeModel):

    def loss_fcn(self, y_true, y_pred, w):
        loss = tf.keras.losses.mse(y_true*w, y_pred*w)
        return loss
            
    # Data pre-proccesing for training, and batch prediction

    def transform(self, data_frame : pd.DataFrame, freq : pd.Timedelta, is_training = False, fill_limit = 3):
        """Resamples and imputate data into fixed timesteps given by ``freq``. If ``is_training = True`` also sets
           mean, std and n_links. Returns resampled data, and imputation mask. """
        travel_time_resample = data_frame.set_index('time').groupby('linkRef')['travelTime'].resample(freq).mean().unstack(0)

        if is_training:
            self.travel_time_mean = travel_time_resample.mean()
            self.travel_time_std = travel_time_resample.std()
        else:
            travel_time_resample = travel_time_resample[travel_time_resample.columns[self.travel_time_mean.index]]

        travel_time_resample_filled = ((travel_time_resample - self.travel_time_mean) / self.travel_time_std).fillna(method='ffill', limit=fill_limit).fillna(method='bfill', limit=fill_limit).dropna(how='all').fillna(0)
        travel_time_resample_mask = travel_time_resample.loc[travel_time_resample_filled.index].notnull()
        
        return travel_time_resample_filled, travel_time_resample_mask

    def split_roll_merge(self, data_frame, lags, preds):
        idx = data_frame.index
        # Splits the data frame at any jumps in the time series
        splits = np.concatenate([[False], pd.to_timedelta(np.diff(idx)) > pd.to_timedelta(self.freq)])
        split_bounds = list(zip(np.concatenate([idx[:1], idx[splits]]), np.concatenate([idx[splits], idx[-1:]])))

        X_lags = []
        T_lags = []
        Y_preds = []
        T_preds = []
        
        # Roll the data
        for split_from, split_to in split_bounds:
            X_lags.append(np.stack([np.roll(data_frame[split_from:split_to].values, shift=n, axis=0) for n in range(lags, 0, -1)], axis=1)[lags:-preds, ...].astype(np.float32))
            T_lags.append(np.stack([np.roll(data_frame[split_from:split_to].index, shift=n, axis=0) for n in range(lags, 0, -1)], axis=1)[lags:-preds, ...])
            Y_preds.append(np.stack([np.roll(data_frame[split_from:split_to].values, shift=-n, axis=0) for n in range(preds)], axis=1)[lags:-preds, ...].astype(np.float32))        
            T_preds.append(np.stack([np.roll(data_frame[split_from:split_to].index, shift=-n, axis=0) for n in range(preds)], axis=1)[lags:-preds, ...])
        # Merge 
        return np.concatenate(X_lags, axis=0), np.concatenate(T_lags, axis=0), np.concatenate(Y_preds, axis=0), np.concatenate(T_preds, axis=0)

    def dow_from_time(self, time, freq):
        time_steps_per_day = int(pd.to_timedelta('24H') / pd.to_timedelta(freq))
        time = pd.to_datetime(time)
        time_step_of_day = pd.to_timedelta([x.isoformat() for x in time.time]) // pd.to_timedelta(freq)
        return (time.weekday * time_steps_per_day + time_step_of_day).astype(int).values

    # Training

    def build_model(self, n_links, freq, lags, preds, params):
        # Parse parameters        
        day_hour_dim = params.get('dayHourDim', 2)
        lr = params.get('lr', 0.01)
        dropout_prop = params.get('dropoutProp', 0.1)
        rnn_hidden_state = params.get('rnnHiddenState', 10)
        
        time_steps_per_day = int(pd.to_timedelta('24H') / pd.to_timedelta(freq))
        
        # Inputs
        in_dow_tod = Input(shape = (lags + preds, ), name = 'dow_tod')
        in_lags = Input(shape = (lags, n_links, ), name = 'lags')
        in_y_true = Input(shape = (preds, n_links, ), name = 'y_true')
        in_mask = Input(shape = (preds, n_links, ), name = 'mask')
        
        # We feed the entire time stampts through the same embeddings layer for coherent learning
        time_embedding = Embedding(7 * time_steps_per_day, day_hour_dim, name='time_embedding')(in_dow_tod)
        time_embedding_lags = tf.keras.layers.Lambda(lambda x: x[:,:lags,:])(time_embedding)
        time_embedding_preds = tf.keras.layers.Lambda(lambda x: x[:,-preds:,:])(time_embedding)
        
        # Pre-processing, encoder input
        bn_lags = tf.keras.layers.BatchNormalization(name='bn_1')(in_lags)
        concat_lags_time = tf.keras.layers.Concatenate(name=f'concat_lags_time')([time_embedding_lags, bn_lags])
        
        # Encoder
        rnn_1 = tf.keras.layers.GRU(rnn_hidden_state, return_sequences=True, return_state=True, unroll=True, name=f'encoder', activation='tanh', dropout=dropout_prop, recurrent_dropout=dropout_prop)
        out_rnn, encoder_state = rnn_1(concat_lags_time)
        out_rnn = tf.keras.layers.Lambda(lambda x: x[:,-preds:,:], name=f'preds')(out_rnn)
        out_rnn = tf.keras.layers.BatchNormalization(name='bn_preds')(out_rnn)
        
        # Decoder
        rnn_2 = tf.keras.layers.GRU(rnn_hidden_state, return_sequences=True, return_state=False, unroll=True, name=f'decoder', activation='tanh', dropout=dropout_prop, recurrent_dropout=dropout_prop)
        out_rnn = rnn_2(out_rnn, initial_state=encoder_state)
        out_rnn = tf.keras.layers.Concatenate(name=f'concat_rnn_time')([time_embedding_preds, out_rnn])
        
        # Dense
        out_rnn = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_links // 3, name='fc_1', activation='relu'), name='time_distributed_fc_1')(out_rnn)        
        out_rnn = Dropout(dropout_prop, name='dropout')(out_rnn)        
        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_links, name='fc_out', activation='linear'), name='time_distributed_output')(out_rnn)

        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        model = Model(inputs = [in_lags,in_dow_tod,in_y_true,in_mask], outputs = [out])
        model.add_loss(self.loss_fcn(in_y_true, out, in_mask))
        model.compile(opt)
        return model
    
    def fit(self, data, labels, params):
        """ The actual training of the ANN. """
        verbose = params.get('verbose', False)
        self.freq = params.get('freq', '15min')
        self.n_lags = params.get('lags', 12)
        self.n_preds = params.get('preds', 4)

        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['linkRef'] = np.array(labels['linkRef'])[df['linkRef']]

        n_pre_outlier = len(df)
        df = self.remove_outlier_mad(df)
        n_post_outlier = len(df)
        n_outlier_pct = (n_pre_outlier - n_post_outlier) / n_pre_outlier
        if verbose:
            print(f'Removed {n_outlier_pct}% outliers ({n_pre_outlier - n_post_outlier}).')

        # Reduce data set for sparse links
        df_train_filled, df_train_mask = self.transform(df, self.freq, is_training = True)
        df_train_filled = df_train_filled.loc[:, (df_train_mask.mean() > .7)]
        df_train_mask = df_train_mask.loc[:, (df_train_mask.mean() > .7)]
        
        self.link_refs = df_train_filled.columns.values
        self.n_links = df_train_filled.shape[1]
        self.travel_time_mean = self.travel_time_mean[self.link_refs]
        self.travel_time_std = self.travel_time_std[self.link_refs]

        self.keras_model = self.build_model(self.n_links, self.freq, self.n_lags, self.n_preds, params)
        
        if verbose:
            print(self.keras_model.summary())

        # Split, roll and merge sequences
        X_train_lags, T_train_lags, Y_train_preds, T_train_preds = self.split_roll_merge(df_train_filled, self.n_lags, self.n_preds)
        W_train_lags, _, W_train_preds, _ = self.split_roll_merge(df_train_mask, self.n_lags, self.n_preds)
        
        # Encode time of day and day of week
        X_train_t_lags = np.zeros_like(T_train_lags, dtype=int)
        for i in range(self.n_lags):
            X_train_t_lags[:,i] = self.dow_from_time(T_train_lags[:,i], self.freq)
        X_train_t_preds = np.zeros_like(T_train_preds, dtype=int)
        for i in range(self.n_preds):
            X_train_t_preds[:,i] = self.dow_from_time(T_train_preds[:,i], self.freq)
        X_train_t = np.concatenate([X_train_t_lags, X_train_t_preds], axis=1)

        fit_callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        ]
        self.history = self.keras_model.fit(
            [X_train_lags,X_train_t,Y_train_preds,W_train_preds],
            [Y_train_preds],
            batch_size=100,
            epochs=100,
            validation_split=.2,
            callbacks=fit_callbacks,
            verbose=verbose)

        # Initialize lags og lag_counts from training data set.
        self.lags = pd.DataFrame(np.concatenate([X_train_lags[-1,-self.n_lags+self.n_preds:,:], Y_train_preds[-1]]), columns=self.link_refs)
        self.lag_counts = pd.DataFrame(np.concatenate([W_train_lags[-1,-self.n_lags+self.n_preds:,:], W_train_preds[-1]]), columns=self.link_refs)
    
    def train(self, time : datetime, spatial_ref : SpatialRef, parameters : Dict[str, Any]) -> Dict[str, Any]:
        """ Implements the train method, which also loads data. """
        verbose = parameters.get('verbose', False)
        n_days = parameters.get('nDays', 21)
        data, data_labels = self.travel_time_n_preceding_normal_days(time, n_days, spatial_ref)

        # Fit the model
        self.fit(data, data_labels, parameters)
        self.time = time
        self.update_predictions()

        return {
            'spatialRefs': self.link_refs.tolist(),
            'trainSamples': len(data['time']),
            'history': self.history.history
        }

    # Event handlers for updating lags and predictions
    def link_completed(self, event_type, event_data):
        link_ref = event_data['linkRef']
        if not link_ref in self.link_refs:
            return
        time = event_data['observedDepartureUtc']
        travel_time = event_data['travelTimeSeconds']

    def update_predictions(self):
        X_t_ = np.array([self.dow_from_time([self.time + x * pd.to_timedelta(self.freq) for x in range(-self.n_lags, self.n_preds)], self.freq)])        
        X_lags_ = np.array([self.lags])
        self.preds = (pd.DataFrame(self.keras_model([X_lags_, X_t_]).numpy()[0], columns=self.link_refs) * self.travel_time_std + self.travel_time_mean).round(1)

    def save(self, model_store, metadata):
        self.keras_model.save(metadata['resourceUrl'])
    
    def restore(self, model_store, metadata):
        _LOGGER.info("Restoring model '%s'...", metadata['ref'])
        # model_store.download_to_cache()
        self.link_refs = np.array(metadata['spatialRefs'])
        # Todo use semaphore - kares unly support one model loading at a time! This will be called from the ThreadPool!
        #self.keras_model = tf.keras.models.load_model(metadata['resourceUrl'])
        self.node.events.listen('linkCompleted', self.link_completed)

    def predict(self, predict_params):
        # Single prediction
        if 'time' in predict_params and isinstance(predict_params['time'], str):
            time = datetime.fromisoformat(predict_params['time'])
            t_steps = pd.to_timedelta(time - self.time) // pd.to_timedelta(self.freq)
            link_ref = predict_params['linkRef']
            
            if not link_ref in self.link_refs or t_steps < -1 or t_steps > self.n_preds + 1:
                return None
            elif t_steps < 0:
                t_steps = 0
            elif t_steps > self.n_preds:
                t_steps = self.n_preds
            
            return self.preds.loc[t_steps, link_ref]
        else:
            raise ValueError('Unsupported predict parameters: %s', predict_params)
