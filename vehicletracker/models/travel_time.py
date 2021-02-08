""" Link Travel Time Models for VehicleTracker """
import logging
from datetime import datetime
from typing import Any, Dict
import json

import numpy as np
import pandas as pd

from vehicletracker.helpers.spatial_reference import SpatialRef
from vehicletracker.models import BaseModel

_LOGGER = logging.getLogger(__name__)

class TravelTimeModel(BaseModel):
    def __init__(self, node):
        self.node = node

    def model_type(self) -> str:
        return 'Link'

    def travel_time_n_preceding_normal_days(self, time, n_days, spatial_ref : SpatialRef):
        """ Service Wrapper for 'travel_time_n_preceding_normal_days' """
        data = self.node.services.call('travel_time_n_preceding_normal_days', {
            'time': time,
            'nDays': n_days,
            'spatialRef': str(spatial_ref)
        })
        if 'error' in data:
            raise ValueError(data['error'])
        return data.get('data'), data.get('labels')

    def travel_time_from_to(self, from_time, to_time, spatial_ref : SpatialRef):
        """ Service Wrapper for 'travel_time_from_to' """
        data = self.node.services.call('travel_time_from_to', {
            'fromTime': from_time,
            'toTime': to_time,
            'spatialRef': str(spatial_ref)
        })
        if 'error' in data:
            raise ValueError(data['error'])
        return data.get('data'), data.get('labels')
