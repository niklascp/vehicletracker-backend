"""Vehicle Tracker Predictor Component"""

import logging
from datetime import datetime, timedelta
import time
from threading import Lock
from typing import (
    Dict,
    List,
    Any,
    Union,
    Tuple
)

import numpy as np
import pandas as pd
import pytz

from vehicletracker.core import VehicleTrackerNode, callback
from vehicletracker.components.model_registry import ModelRegitry, METADATA_ATTR_MODEL_NAME, METADATA_ATTR_MODEL_REF
from vehicletracker.components.model_registry import DOMAIN as MODEL_REGISTRY
from vehicletracker.exceptions import ModelNotFound
from vehicletracker.helpers.datetime import DEFAULT_TIME_ZONE

DOMAIN = 'predictor'

_LOGGER = logging.getLogger(__name__)

MODEL_CACHE_PATH = './cache/models/'

async def async_setup(node : VehicleTrackerNode, config : Dict[str, Any]):    
    """Sets up the predictor component"""

    model_registry = node.data[MODEL_REGISTRY] 
    predictor = node.data[DOMAIN] = Predictor(node, model_registry, config[DOMAIN])
    
    # Wire up events and services
    await node.services.async_register(DOMAIN, 'link_predict', predictor.link_predict)
    await node.services.async_register(DOMAIN, 'stop_predict', predictor.stop_predict)

    await node.events.async_listen('model_available', predictor.model_available)

    return True

class Predictor():
    """Predictor State"""

    def __init__(self, node : VehicleTrackerNode, model_registry : ModelRegitry, config : Dict[str, Any]):
        self.model_registry = model_registry
        self.model_cache : Dict[str, ] = {}
        self.spatial_latest_map : Dict[str, List[str]] = {}
        self.lock = Lock()

    def decode_temporal_ref(self, serviceData) -> Tuple[datetime, List[datetime]]:
        time = serviceData.get('time')

        if isinstance(time, str):
            time_min = datetime.fromisoformat(time)
            time_ = [time_min]
        elif isinstance(time, list):
            if all(isinstance(x, int) for x in time):
                time_ = pd.to_datetime(time, unit='s')
            elif all(isinstance(x, str) for x in time):
                time_ = pd.to_datetime(time)
            else:
                raise ValueError('if time is a list, all items must be either string or integers.')
            time_min = time_.min()
        else:
            time_min = datetime.now()
            time_ = [time_min]
        
        if time_min.tzinfo is None:
            time_min = DEFAULT_TIME_ZONE.localize(time_min)

        return time_min, time_

    def get_model(self, model_metadata):
        # Load from memory
        model_ref = model_metadata[METADATA_ATTR_MODEL_REF]
        if model_ref in self.model_cache:
            return self.model_cache[model_ref]
        
        #TODO: Listen for model_available to clear and relead cache
        #if model_ref in self.model_cache:
        #    del self.model_cache[model_ref]
        self.lock.acquire()
        try:
            if model_ref in self.model_cache:
                return self.model_cache[model_ref]
            self.model_cache[model_ref] = self.model_registry.get_model(model_metadata)
            return self.model_cache[model_ref]
        finally:
            self.lock.release()

    def link_predict(self, service_data):
        """Service handler for 'link_predict'. Predicts a single link at one or more points in time."""

        link_ref = service_data['linkRef']
        model_name = service_data.get('model')
        
        temporal_ref, time = self.decode_temporal_ref(service_data)
        spatial_ref = link_ref         
        
        _LOGGER.debug("Recived link predict request for link '%s' using model '%s' at time %s.",
            link_ref, model_name, temporal_ref)

        temporal_ref_utc = temporal_ref.astimezone(pytz.utc).replace(tzinfo=None)
        latest_horizon_utc = datetime.utcnow() - timedelta(minutes=1)

        # This is normal opreration, i.e. predicting into the future, always use *latest* model cached in spatial_latest_map.
        if temporal_ref_utc > latest_horizon_utc:
            model_candidates = self.spatial_latest_map.get(spatial_ref, None)
            if model_candidates is None:
                model_candidates = self.model_registry.model_store.list_models(model_name, spatial_ref, temporal_ref)
                self.spatial_latest_map[spatial_ref] = model_candidates
        # This is model developper operation, i.e. historic prediction, look for most recent model at the time.
        else:
            model_candidates = self.model_registry.model_store.list_models(model_name, spatial_ref, temporal_ref)
            # Rewrite stopPointRef for batch prediction.
            if len(time) > 1:
                service_data['linkRef'] = [link_ref] * len(time)

        if len(model_candidates) == 0:
            _LOGGER.warning("No model candidates for link '%s' using model '%s' at time %s was found.", spatial_ref, model_name or 'ANY', temporal_ref)
            return []

        preds = []
        for model_metadata in model_candidates:
            model = self.get_model(model_metadata)
            pred = model.predict(service_data)
            if pred is not None:
                preds.append({
                    'model': model_metadata[METADATA_ATTR_MODEL_NAME],
                    'model_ref': model_metadata[METADATA_ATTR_MODEL_REF],
                    'predicted': np.round(np.squeeze(pred), 1).tolist()
                })

        return preds

    def stop_predict(self, service_data):
        """Service handler for 'stop_predict'. Predicts dwll time for a single stop at one or more points in time."""

        stop_point_ref = service_data.get('stopPointRef')
        model_name = service_data.get('model')

        temporal_ref, at_time = self.decode_temporal_ref(service_data)
        spatial_ref = stop_point_ref 

        _LOGGER.debug("Recived stop predict request for stop '%s' using model '%s' at time %s.",
            spatial_ref, model_name , temporal_ref)

        temporal_ref_utc = temporal_ref.astimezone(pytz.utc).replace(tzinfo=None)
        latest_horizon_utc = datetime.utcnow() - timedelta(minutes=1)

        # This is normal opreration, i.e. predicting into the future, always use *latest* model cached in spatial_latest_map.
        if temporal_ref_utc > latest_horizon_utc:
            model_candidates = self.spatial_latest_map.get(spatial_ref, None)
            if model_candidates is None:
                model_candidates = self.model_registry.model_store.list_models(model_name, spatial_ref, temporal_ref)
                self.spatial_latest_map[spatial_ref] = model_candidates
        # This is model developper operation, i.e. historic prediction, look for most recent model at the time.
        else:
            model_candidates = self.model_registry.model_store.list_models(model_name, spatial_ref, temporal_ref)
            # Rewrite stopPointRef for batch prediction.
            if len(at_time) > 1:
                service_data['stopPointRef'] = [stop_point_ref] * len(at_time)


        if len(model_candidates) == 0:
            _LOGGER.info("No model condidates for stop '%s' using model '%s' at time %s was found.", stop_point_ref, model_name or 'ANY', temporal_ref)
            return []

        preds = []
        for model_metadata in model_candidates:
            model = self.get_model(model_metadata)
            start_time = time.time()
            pred = model.predict(service_data)
            predict_time = time.time() - start_time

            if predict_time > 1:
                _LOGGER.warning("Prediction for stop '%s' using model '%s' took %ss.", stop_point_ref, model_metadata[METADATA_ATTR_MODEL_REF], np.round(predict_time, 1)) 

            if pred is not None:
                preds.append({
                    'model': model_metadata[METADATA_ATTR_MODEL_NAME],
                    'model_ref': model_metadata[METADATA_ATTR_MODEL_REF],
                    'predicted': np.round(np.squeeze(pred), 1).tolist(),
                    #'predict_time': predict_time
                })

        return preds

    def model_available(self, event_name, event_data): 
        pass