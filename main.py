import sys
import os
import logging
import threading
import signal

from vehicletracker.helpers.config import load_config
from vehicletracker.helpers.logging import setup_logging

from vehicletracker.components import history_local
from vehicletracker.components import trainer
from vehicletracker.components import predictor
from vehicletracker.components import recorder

_LOGGER = logging.getLogger()

if __name__ == '__main__':

    def signal_handler(signal, frame):
        _LOGGER.info("stop signal recieved")
        stop_event.set()

    config = load_config(sys.argv[1:])
    setup_logging(config)
    stop_event = threading.Event()
    
    signal.signal(signal.SIGINT, signal_handler)

    # TODO: Clearly this cam be made more buetifully
    component_map = {
        'history_local': history_local,
        'trainer': trainer,
        'predictor': predictor,
        'recorder': recorder
    }

    components = [v for k, v in component_map.items() if '-component:' + k in sys.argv]
    component_names = [k for k, v in component_map.items() if '-component:' + k in sys.argv]

    _LOGGER.info(f"staring components {component_names} ...")

    for component in components:
        try:
            component.start()
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception(f"Error during setup of component {component}")

    while not stop_event.is_set():
        stop_event.wait(1) 

    for component in components:
        component.stop()        
