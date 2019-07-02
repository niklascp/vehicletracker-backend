import sys
import os
import logging
from logging.handlers import TimedRotatingFileHandler

from vehicletracker.consts import EVENTS_EXCHANGE_NAME
from vehicletracker.consts.events import EVENT_LINK_PREDICT_REQUEST, EVENT_LINK_PREDICT_COMPLETED, EVENT_LINK_TRAIN_COMPLETED

import pika
import json
from datetime import datetime

import pandas as pd

WORKER_QUEUE_NAME = 'vehicletracker-predictor-events'
LISTEN_EVENT_TYPES = [EVENT_LINK_PREDICT_REQUEST, EVENT_LINK_TRAIN_COMPLETED]

_LOGGER = logging.getLogger(__name__)

LINK_MODELS = {}

from typing import Any, Callable

import threading

class EventQueue():    

    def __init__(self):
        credentials = pika.PlainCredentials(
            username = os.environ['RABBITMQ_USERNAME'],
            password = os.environ['RABBITMQ_PASSWORD'])

        parameters = pika.ConnectionParameters(
            host = os.getenv('RABBITMQ_ADDRESS', 'localhost'),
            port = os.getenv('RABBITMQ_PORT', 5672),
            virtual_host = os.getenv('RABBITMQ_VHOST', '/'),
            credentials = credentials)

        # Declare connections and channel
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        # Declare worker queue
        self.channel.queue_declare(queue = WORKER_QUEUE_NAME, auto_delete = True)

        # Declare client specific request/response queue
        result = self.channel.queue_declare(queue = '', exclusive = True)
        self.callback_queue = result.method.queue
        
        # Bind request/response queue
        self.channel.queue_bind(exchange = EVENTS_EXCHANGE_NAME, queue = self.callback_queue, routing_key = '#')

        # Hook up consumers
        self.channel.basic_consume(queue = WORKER_QUEUE_NAME, on_message_callback = self.event_callback, auto_ack = True)
        self.channel.basic_consume(queue = self.callback_queue, on_message_callback = self.reply_callback, auto_ack = True)

        self.thread = threading.Thread(target=self.channel.start_consuming)
        self.listeners = {}

    def publish_event(self, event, reply_to = None, correlation_id = None):
        self.channel.basic_publish(
            exchange = EVENTS_EXCHANGE_NAME,
            routing_key = event['eventType'],
            body = json.dumps(event),
            properties = pika.BasicProperties (
                content_type = 'application/json',
                correlation_id = correlation_id
            )
        )

        if reply_to is not None:
            try:
                self.channel.basic_publish(
                    exchange = '',
                    routing_key = reply_to,
                    body = json.dumps(event),
                    properties = pika.BasicProperties (
                        content_type = 'application/json',
                        correlation_id = correlation_id
                    )
                )
            except Exception:
                _LOGGER.exception('Could not reply error:')

    def event_callback(self, ch, method, properties, body):
        try:
            if properties.content_type == 'application/json':
                event = json.loads(body)
                event_type = event['eventType']

                target_list = self.listeners.get(event_type, [])

                _LOGGER.debug(f"handling {event_type} with {len(target_list)} targets.")

                for target in target_list:
                    # TODO: Wrap targets in exception logging and execute async.
                    try:
                        #task_runner.async_add_job(target, *args)
                        result_event = target(event)

                        if result_event:
                            self.publish_event(
                                event = result_event,
                                reply_to = properties.reply_to,
                                correlation_id = properties.correlation_id)

                    except Exception as e:
                        _LOGGER.exception('Error in executing listerner.')
                        self.publish_event(
                            event = {
                                'eventType': 'error',
                                'error': str(e)
                            },
                            reply_to = properties.reply_to,
                            correlation_id = properties.correlation_id
                        )

        except Exception:
            _LOGGER.exception('Error in event_callback.')
    
    def reply_callback(self, ch, method, properties, body):
        try:
            if properties.content_type == 'application/json':
                event = json.loads(body)
                correlation_id = properties.correlation_id

        except Exception:
            _LOGGER.exception('Error in reply_callback.')

    def listen(self, event_type: str, target: Callable[..., Any]) -> Callable[[], None]:
        _LOGGER.info(f"listening for {event_type}.")
        if event_type not in self.listeners:
            self.listeners[event_type] = []
            self.channel.queue_bind(
                exchange = EVENTS_EXCHANGE_NAME,
                queue = WORKER_QUEUE_NAME,
                routing_key = event_type)

        self.listeners[event_type].append(target)
        
    def start(self):
        self.thread.start()


event_queue = EventQueue()

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

    event_queue.listen(EVENT_LINK_PREDICT_REQUEST, predict)
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
