import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

class WeeklySvr:
    
    def __init__(self, grid_search = True, verbose = True):
        self.freq = pd.to_timedelta('1min')
        self.timesteps_per_day = pd.to_timedelta('24H') / self.freq
        self.grid_search = grid_search
        self.verbose = verbose
    
    def transform_ix(self, ix):
        """ Map time into an integer interval $\mathbb{R}^2 \in [0; 2 \pi[$ representing the time of day and day of week with 1 minute granuality. """        
        x = ix.dayofweek.values * self.timesteps_per_day + ix.map(lambda x: pd.to_timedelta(x.time().isoformat())).values / self.freq
        x_2 = ix.map(lambda x: pd.to_timedelta(x.time().isoformat())).values / self.freq
        X = np.stack([
            np.sin(x / (7*24*60) * 2 * np.pi),
            np.cos(x / (7*24*60) * 2 * np.pi),
            np.sin(x_2 / (24*60) * 2 * np.pi),
            np.cos(x_2 / (24*60) * 2 * np.pi)
        ], axis = 1)
        return X
        
    def fit(self, ix, Y):
        X = self.transform_ix(ix)        
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        elif len(Y.shape) > 2:
            raise ValueError("Expected Y of two dimensions, but Y has {} dimensions".format(len(Y.shape)))
       
        self.models = []
        for i in range(Y.shape[1]):
            include = ~np.isnan(Y[:, i])
            
            parameters = {
                'C': np.logspace(0, 4, 5),
                'gamma': np.logspace(-2, 2, 5),
                'epsilon': np.logspace(-5, -1, 5)
            }
            svr = SVR(verbose = self.verbose)
            clf = GridSearchCV(svr, parameters, cv=5, n_jobs = -1, verbose = self.verbose)
            
            clf.fit(X[include, ], Y[include, i])
                        
            self.models.append(clf) 
            
    def predict(self, ix):
        X = self.transform_ix(ix)
        return np.stack([
            clf.predict(X)
            for clf in self.models
        ], axis = 1)
