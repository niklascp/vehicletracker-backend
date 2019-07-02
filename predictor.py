import sys
import os
import logging
from logging.handlers import TimedRotatingFileHandler

from vehicletracker.consts.events import EVENT_LINK_PREDICT_REQUEST, EVENT_LINK_PREDICT_COMPLETED, EVENT_LINK_TRAIN_COMPLETED
from vehicletracker.data.events import EventQueue

from datetime import datetime

import json

import pandas as pd

_LOGGER = logging.getLogger('vehicletraker.predictor')

LINK_MODELS = {}

event_queue = EventQueue(worker_queue_name = 'predictor')

def predict(event):
    link_ref = event['linkRef']
    time = datetime.fromisoformat(event.get('time')) if event.get('time') else datetime.now()
    model_name = event['model']
    model = LINK_MODELS[link_ref]['model']

    _LOGGER.debug(f"Recived link predict request for link '{link_ref}' using model '{model_name}' at time {time}.")

    ix = pd.DatetimeIndex([time])
    pred = model.predict(ix)

    return {
        'eventType': EVENT_LINK_PREDICT_COMPLETED,
        'linkRef': link_ref,
        'time': time.isoformat(),
        'prediction': pred[0, 0]
    }

def main():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logHandler = TimedRotatingFileHandler("./logs/predictor.log", when="midnight")
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logHandler.setFormatter(logFormatter)
    _LOGGER.addHandler(logHandler)
    _LOGGER.setLevel(logging.DEBUG)

    for l_ in ['vehicletracker.data.events']:
        l = logging.getLogger(l_)
        l.addHandler(logHandler)
        l.setLevel(logging.DEBUG)


    # TODO: Move into class DiskModelCache
    import joblib
    _LOGGER.info('Loading cached models from disk ...')
    MODEL_CACHE_PATH = './cache/lt-link-travel-time/'
    if not os.path.exists(MODEL_CACHE_PATH):
        os.makedirs(MODEL_CACHE_PATH)
    for file_name in os.listdir(MODEL_CACHE_PATH):
        if (file_name.endswith(".json")):
            _LOGGER.info(f'Loading cached model from data from {file_name}')
            metadata_file_path = os.path.join(MODEL_CACHE_PATH, file_name)
            model_file_path = os.path.splitext(os.path.join(MODEL_CACHE_PATH, file_name))[0] + '.joblib'
            
            with open(metadata_file_path, 'r') as f:
                model_metadata = json.load(f)
            with open(model_file_path, 'rb') as f:
                model = joblib.load(f)
            
            LINK_MODELS[model_metadata['linkRef']] = { 'model': model, 'metadata': model_metadata }

    #event_queue.listen(EVENT_LINK_PREDICT_REQUEST, predict)
    event_queue.register_service('link.predict', predict)
    event_queue.start()

    _LOGGER.info('Predictor ready, waiting for events.')
    print(' [*] Waiting for events. To exit press CTRL+C')

    input("Press Enter to exit...")
    os._exit(0)

    """
    import signal
    import sys
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    """

if __name__ == '__main__':
    main()
