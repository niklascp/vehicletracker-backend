"""Tests for the model_registry component."""

import pytest

from vehicletracker.components import model_registry
from vehicletracker.components import trainer
from vehicletracker.components import predictor
from vehicletracker.core import VehicleTrackerNode, async_setup_component, _LOGGER

@pytest.mark.asyncio
async def test_mock_model():
    """Set up things to be run when tests are started."""
    CONFIG = {
        model_registry.DOMAIN: {
            model_registry.ATTR_MODELS: [
                {
                    model_registry.ATTR_NAME: 'MockModel',
                    model_registry.ATTR_CLASS: 'tests.models.test_models.MockModel'
                }
            ]
        },
        trainer.DOMAIN: {},
        predictor.DOMAIN: {}
    }
    node = VehicleTrackerNode(CONFIG)
    assert await async_setup_component(node, model_registry.DOMAIN, CONFIG) == True
    assert await async_setup_component(node, trainer.DOMAIN, CONFIG) == True
    assert await async_setup_component(node, predictor.DOMAIN, CONFIG) == True

    await node.async_start()

    job = await node.services.async_call('schedule_train_model', {
        'model': 'MockModel',
        'time': '2020-07-28 00:00:00',
        'parameters': {}
    }, timeout=1)

    assert job['jobId'] == 1
    
    await node.async_stop()
