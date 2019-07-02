import numpy as np
import pandas as pd

from scipy import interpolate

class WeeklyKernelInterpolation:
    
    def __init__(self, smooth):
        self.freq = pd.to_timedelta('1min')
        self.timesteps_per_day = pd.to_timedelta('24H') / self.freq
        self.smooth = smooth
    
    def norm(self, XA, XB):
        """ Calculates the distance between to points in the circular time. """
        d1 = abs(XA - XB)
        d2 = abs(XA - XB - 7*24*60)
        return np.vstack([d1, d2]).min(axis = 0)
    
    def transform_ix(self, ix):
        """ Map time into an integer interval [0; 7 * 24 * 60[ representing the time of day and day of week with 1 minute granuality. """        
        return (ix.dayofweek.values * self.timesteps_per_day + ix.map(lambda x: pd.to_timedelta(x.time().isoformat())).values / self.freq)
        
    def fit(self, ix, Y):
        x = self.transform_ix(ix)        
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        elif len(Y.shape) > 2:
            raise ValueError("Expected Y of two dimensions, but Y has {} dimensions".format(len(Y.shape)))
       
        self.models = []
        for i in range(Y.shape[1]):
            include = ~np.isnan(Y[:, i])
            rbf = interpolate.Rbf(x[include], Y[include, i], smooth = self.smooth, norm = self.norm)
            self.models.append(rbf) 
            
    def predict(self, ix):
        x = self.transform_ix(ix)
        return np.stack([
            rbf(x)
            for rbf in self.models
        ], axis = 1)
    