import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class WeeklyHistoricalAverage:
    
    def __init__(self, freq = 'auto', verbose = True):
        self.freq = freq
        self.verbose = verbose
    
    def transform_ix(self, ix, freq):
        """ Map time into an integer interval [0; 7 * (24H/freq)[ representing the time of day and day of week with freq granuality. """
        timesteps_per_day = pd.to_timedelta('24H') / freq
        return (ix.dayofweek.values * timesteps_per_day + pd.to_timedelta(ix.hour * 60 * 60 + ix.minute * 60 + ix.second, unit='s') // freq).astype(int)
        
    def fit(self, ix, Y):
        if self.freq == 'auto':
            freqs = ['5min', '10min', '15min', '30min', '1H']
        else:
            freqs = [self.freq]
            
        kf = KFold(n_splits=5)

        self.loss = None
        
        for freq in freqs:            
            freq_ = pd.to_timedelta(freq)            
            x = self.transform_ix(ix, freq_)
            time = pd.date_range(pd.to_datetime('1970-01-01'), pd.to_datetime('1970-01-08'), freq = freq_, closed='left')
            
            loss_ = []
            for ix_train, ix_test in kf.split(ix):
                df_ha = pd.DataFrame(index = x[ix_train] * freq_, data = Y[ix_train])
                # Pandas resampel only works with datetimes - set some reference data
                df_ha.index = pd.to_datetime('1970-01-01') + df_ha.index
                model = df_ha.resample(freq_).mean().reindex(time).interpolate().fillna(method = 'ffill').fillna(method = 'bfill')
            
                loss_.append(np.mean((Y[ix_test] - self.predict_(model, freq_, x[ix_test]))**2))
            
            print(freq, 'loss:', loss_)
            
            if self.loss == None or np.mean(loss_) < self.loss:
                self.model = model
                self.freq_selected = freq
                self.loss = np.mean(loss_)
    
        print('selected:', self.freq_selected)
            
    
    def predict_(self, model, freq, x):
        df_ha = pd.DataFrame(index = x * freq)
        df_ha.index = pd.to_datetime('1970-01-01') + df_ha.index
        return model.loc[df_ha.index].values
    
    def predict(self, ix):
        freq = pd.to_timedelta(self.freq_selected)
        x = self.transform_ix(ix, freq)
        return self.predict_(self.model, freq, x)
