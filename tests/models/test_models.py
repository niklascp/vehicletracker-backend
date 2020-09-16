from datetime import datetime
from typing import Callable, Dict, Optional

from vehicletracker.helpers.spatial_reference import parse_spatial_ref
from vehicletracker.models import BaseModel
from vehicletracker.models.travel_time import WeeklyHistoricalAverageTravelTime


class MockNode():

    def __init__(self):
        self.services = MockService()

class MockService():

    def __init__(self):
        self.map : Dict[str, Callable] = {}

    def call(self, service: str, service_data : Optional[Dict] = None):
        return self.map[service](service_data)

    def register(self, domain: str, service: str, service_func: Callable):
        self.map[service] = service_func

class MockModel(BaseModel):

    def __init__(self, node):
        pass

    def model_type(self) -> str:
        return 'Test'

    def train(self, parameters):
        return {
            'type': 'mock',
            'spatialRefs': 'mock'
        }

    def save(self, model_store, metadata):
        pass
    
    def restore(self, model_store, metadata):
        pass

def test_weekly_historical_average_travel_time():
    def mock_travel_time(service_data):
        """This mocks a reply for 'travel_time_n_preceding_normal_days' from history component"""
        return {
            'labels': {
                'link_ref': ['A:B', 'B:C']
            },
            'data': {
                'time': [1577836800, 1577836800, 1577836800, 1577840400, 1577840400, 1577840400],
                'link_ref': [0, 0, 0, 1, 1, 1],
                'link_travel_time': [10, 8, 12, 20, 18, 22]
            }
        }

    mock_node = MockNode()
    mock_node.services.register('history', 'travel_time_n_preceding_normal_days', mock_travel_time)
    model = WeeklyHistoricalAverageTravelTime(mock_node)
    result = model.train(datetime.fromisoformat('2020-07-28 00:00'), parse_spatial_ref('L:1'), {})
    #assert result['loss'] == 4

    model_metadata = {
        'ref': 'test',
        'resourceUrl': './test_cache/'
    }

    predict_params = { 'time': '2020-01-01 00:00:00', 'linkRef': 'A:B' }

    assert model.predict(predict_params) == 10

    model.save(None, model_metadata)
    model_restored = WeeklyHistoricalAverageTravelTime(mock_node)
    model_restored.restore(None, model_metadata)

    assert model.predict(predict_params) ==  model_restored.predict(predict_params)
