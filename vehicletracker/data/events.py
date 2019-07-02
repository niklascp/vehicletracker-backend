import os
import logging
import threading

import uuid

import pika
import json

from vehicletracker.consts import EVENTS_EXCHANGE_NAME

from typing import Any, Callable

_LOGGER = logging.getLogger(__name__)

class EventQueue():    

    def __init__(self, worker_queue_name):
        self.worker_queue_name = worker_queue_name

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

        # Declare exchange / queue for service calls
        # self.channel.exchange_declare(exchange = REQUESTS_EXCHANGE_NAME, exchange_type = 'topic')
        # self.channel.queue_declare(queue = REQUESTS_QUEUE_NAME, auto_delete = True)
        
        # Declare exchange / queue for events
        self.channel.exchange_declare(exchange = EVENTS_EXCHANGE_NAME, exchange_type = 'topic')
        self.channel.queue_declare(queue = self.worker_queue_name, auto_delete = True)
        
        # Declare queue for service callbacks
        result = self.channel.queue_declare(queue = '', exclusive = True)
        self.callback_queue = result.method.queue
        
        # Hook up consumers
        self.channel.basic_consume(queue = self.worker_queue_name, on_message_callback = self.event_callback, auto_ack = True)
        self.channel.basic_consume(queue = self.callback_queue, on_message_callback = self.reply_callback, auto_ack = True)

        self.thread = threading.Thread(target=self.channel.start_consuming)
        self.services = {}
        self.listeners = {}

    def publish_event(self, event, reply_to = None, correlation_id = None):
        self.channel.basic_publish(
            exchange = EVENTS_EXCHANGE_NAME,
            routing_key = event['eventType'],
            body = json.dumps(event),
            properties = pika.BasicProperties (
                content_type = 'application/json',
                reply_to = reply_to,
                correlation_id = correlation_id
            )
        )

    def publish_reply(self, data, to = None, correlation_id = None):
        self.channel.basic_publish(
            exchange = '',
            routing_key = to,
            body = json.dumps(data),
            properties = pika.BasicProperties (
                content_type = 'application/json',
                correlation_id = correlation_id
            )
        )

    def event_callback(self, ch, method, properties, body):
        try:
            if properties.content_type == 'application/json':
                event = json.loads(body)
                event_type = event['eventType']

                target_list = self.listeners.get(event_type, [])

                _LOGGER.debug(f"handling {event_type} with {len(target_list)} targets (reply_to: {properties.reply_to}, correlation_id: {properties.correlation_id})")

                for target in target_list:
                    # TODO: Wrap targets in exception logging and execute async.
                    try:
                        #task_runner.async_add_job(target, *args)
                        result_data = target(event)

                        if properties.reply_to:
                            self.publish_reply(
                                data = result_data,
                                to = properties.reply_to,
                                correlation_id = properties.correlation_id)

                    except Exception as e:
                        _LOGGER.exception('error in executing listerner')
                        self.publish_reply(
                            data = {
                                'error': str(e)
                            },
                            to = properties.reply_to,
                            correlation_id = properties.correlation_id)
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
                queue = self.worker_queue_name,
                routing_key = event_type)

        self.listeners[event_type].append(target)
    
    def register_service(self, service_name, target):
        #self.listeners[f'service.{name}'] = []
        self.listen(f'service.{service_name}', lambda x: target(x.get('serviceData')))

    def call_service(self, service_name, service_data, timeout = 1000):
        correlation_id = str(uuid.uuid4())
        self.publish_event(
            event = {
                'eventType': f'service.{service_name}',
                'serviceData': service_data
            },
            reply_to = self.callback_queue
            correlation_id = correlation_id
        )

    def start(self):
        self.thread.start()
