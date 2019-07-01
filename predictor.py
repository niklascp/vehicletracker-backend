import os
import logging
from logging.handlers import TimedRotatingFileHandler

from vehicletracker.consts.events import EVENT_LINK_PREDICT_REQUEST, EVENT_LINK_TRAIN_COMPLETED

import pika
import json

EVENTS_EXCHANGE_NAME = 'vehicletracker-events'
TRAINER_WORKER_QUEUE_NAME = 'vehicletracker-predictor-events'
LISTEN_EVENT_TYPES = [EVENT_LINK_PREDICT_REQUEST, EVENT_LINK_TRAIN_COMPLETED]

_LOGGER = logging.getLogger(__name__)

LINK_MODELS = {}

def callback(ch, method, properties, body):
    msg = json.loads(body)
    event_type = msg['eventType']
    if event_type == EVENT_LINK_PREDICT_REQUEST:
        link_ref = msg['linkRef']
        time = pd.to_datetime(msg.get('time') or pd.datetime.now())
        model_name = msg['model']
        
        _LOGGER.debug(f"Recived *link predict request* for link '{link_ref}' using model '{model_name}' at time {time}.")


    else:
        _LOGGER.error(f'Unknown message: {event_type}.')

def main():
    logHandler = TimedRotatingFileHandler("logs/predictor.log", when="midnight")
    _LOGGER.addHandler(logHandler)
    _LOGGER.setLevel(logging.DEBUG)

    credentials = pika.PlainCredentials(
        username = os.environ['RABBITMQ_USERNAME'],
        password = os.environ['RABBITMQ_PASSWORD'])

    parameters = pika.ConnectionParameters(
        host = os.getenv('RABBITMQ_ADDRESS', 'localhost'),
        port = os.getenv('RABBITMQ_PORT', 5672),
        virtual_host = os.getenv('RABBITMQ_VHOST', '/'),
        credentials = credentials)

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    channel.queue_declare(queue=TRAINER_WORKER_QUEUE_NAME, auto_delete=True)

    for event_type in LISTEN_EVENT_TYPES:
        channel.queue_bind(exchange=EVENTS_EXCHANGE_NAME, queue=TRAINER_WORKER_QUEUE_NAME, routing_key=event_type)

    # TODO: Move into class DiskModelCache
    import joblib
    _LOGGER.info('Loading cached models from disk ...')
    MODEL_CACHE_PATH = './models/lt-link-travel-time/'
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


    _LOGGER.info('Predictor ready, waiting for events.')
    print(' [*] Waiting for events. To exit press CTRL+C')

    channel.basic_consume(
        queue=TRAINER_WORKER_QUEUE_NAME,
        on_message_callback=callback,
        auto_ack=True)

    channel.start_consuming()

if __name__ == '__main__':
    main()
