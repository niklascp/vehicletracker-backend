import os
import logging
from logging.handlers import TimedRotatingFileHandler

from vehicletracker.consts.events import EVENT_LINK_TRAIN_REQUEST, EVENT_LINK_TRAIN_COMPLETED

import pika
import json

EVENTS_EXCHANGE_NAME = 'vehicletracker-events'
TRAINER_WORKER_QUEUE_NAME = 'vehicletracker-trainer-events'
LISTEN_EVENT_TYPES = [EVENT_LINK_TRAIN_REQUEST]

KNOWN_MODELS = {
    'svr': None
}

_LOGGER = logging.getLogger(__name__)

def callback(ch, method, properties, body):
    msg = json.loads(body)
    event_type = msg['eventType']
    if event_type == EVENT_LINK_TRAIN_REQUEST
        link_ref = msg['linkRef']
        model_name = msg['model']
        model_parameters = msg['parameters']
        _LOGGER.debug(f"Recived *link train request* for link '{link_ref}' using model '{model_name}'.")

        # TODO: Should be moved into a class.        
        import pandas as pd

        from vehicletracker.data.client import PostgresClient
        from vehicletracker.models import WeeklySvr

        import joblib

        data = PostgresClient()

        n = model_parameters['n']
        time = pd.to_datetime(model_parameters.get('time') or pd.datetime.now())
        train = data.link_travel_time_n_preceding_normal_days(link_ref, time, n)
        _LOGGER.debug(f"Loaded train data: {train.shape[0]}")

        weekly_svr = WeeklySvr()
        #weekly_svr.fit(train.index, train.values)

        link_ref_safe = link_ref.replace(':', '-')
        MODEL_CACHE_PATH = './models/lt-link-travel-time/'

        metadata_file_name = f'{model_name}-{time:%Y%m%d}-{n}-{link_ref_safe}.json'
        model_file_name = f'{model_name}-{time:%Y%m%d}-{n}-{link_ref_safe}.joblib'

        metadata = {
            'model': model_name,
            'linkRef': link_ref,
            'time': time.isoformat(),
        }

        # Write metadata
        with open(os.path.join(MODEL_CACHE_PATH, metadata_file_name), 'w') as f:
            json.dump(f, metadata)
        
        # Write model
        joblib.dump(weekly_svr, os.path.join(MODEL_CACHE_PATH, model_file_name))

    else:
        _LOGGER.error('Unknown message.')

def main():
    logHandler = TimedRotatingFileHandler("logs/trainer.log", when="midnight")
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logHandler.setFormatter(logFormatter)
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

    _LOGGER.info('Worker ready, waiting for events.')
    print(' [*] Waiting for events. To exit press CTRL+C')

    channel.basic_consume(
        queue=TRAINER_WORKER_QUEUE_NAME,
        on_message_callback=callback,
        auto_ack=True)

    channel.start_consuming()

if __name__ == '__main__':
    main()
