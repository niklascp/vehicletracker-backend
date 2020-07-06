"""Tests for the model_registry component."""

import pytest

from vehicletracker.components import model_registry
from vehicletracker.core import VehicleTrackerNode, async_setup_component

@pytest.mark.asyncio
async def test_mock_model():
    """Set up things to be run when tests are started."""
    CONFIG = {
        model_registry.DOMAIN: {
            model_registry.ATTR_MODELS: [
                {
                    model_registry.ATTR_NAME: 'MockModel',
                    model_registry.ATTR_CLASS: __name__
                }
            ]
        }
    }
    node = VehicleTrackerNode(CONFIG) #get_test_home_assistant()
    assert await async_setup_component(node, model_registry.DOMAIN, CONFIG) == True

    assert len(node.data[model_registry.DOMAIN].models) == 1

