from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from vehicletracker.helpers.spatial_reference import SpatialRef

class BaseModel(ABC):
    def __init__(self, node):
        self.node = node

    @abstractmethod
    def model_type(self) -> str:
        pass

    @abstractmethod
    def train(self, time : datetime, spatial_ref : SpatialRef, parameters : Dict[str, Any]) -> Dict[str, Any]:
        pass
