import os
import logging
import socket
import threading

import time
import uuid

import pika
import json

from typing import Any, Callable

#REQUESTS_EXCHANGE_NAME = 'vehicletracker-requests'
EVENTS_EXCHANGE_NAME = 'vehicletracker-events'

_LOGGER = logging.getLogger(__name__)

class EventQueue():    

    def __init__(self, domain):
        self.worker_queue_name = domain + '-worker'
        self.callback_queue = domain + '-client-' + socket.gethostname()

        credentials = pika.PlainCredentials(
            username = os.environ['RABBITMQ_USERNAME'],
            password = os.environ['RABBITMQ_PASSWORD'])

        self.parameters = pika.ConnectionParameters(
            host = os.getenv('RABBITMQ_ADDRESS', 'localhost'),
            port = os.getenv('RABBITMQ_PORT', 5672),
            virtual_host = os.getenv('RABBITMQ_VHOST', '/'),
            credentials = credentials)
        
        self.thread = None
        self.cancel_event = None        

        self.consume_connection = None
        self.consume_channel = None

        self.connection = None
        self.channel = None

        self.wait_events = {}
        self.results = {}
        self.listeners = {}

    def consume_internal(self):
        while not self.cancel_event.is_set():
            try:
                _LOGGER.info('worker connecting to RabbitMQ ...')

                # Declare connections and channel
                self.consume_connection = pika.BlockingConnection(self.parameters)
                self.consume_channel = self.consume_connection.channel()

                # Declare exchange / queue for service calls
                # self.channel.exchange_declare(exchange = REQUESTS_EXCHANGE_NAME, exchange_type = 'topic')
                # self.channel.queue_declare(queue = REQUESTS_QUEUE_NAME, auto_delete = True)
                        
                # Declare exchange / queue for events
                self.consume_channel.exchange_declare(exchange = EVENTS_EXCHANGE_NAME, exchange_type = 'topic')
                self.consume_channel.queue_declare(queue = self.worker_queue_name, auto_delete = True)
                
                # Declare queue for service callbacks        
                self.consume_channel.queue_declare(queue = self.callback_queue, exclusive = True)
                #result = self.channel.queue_declare(queue = '', exclusive = True)
                #self.callback_queue = result.method.queue
        
                # Hook up consumers
                self.consume_channel.basic_consume(queue = self.worker_queue_name, on_message_callback = self.event_callback, auto_ack = True)
                self.consume_channel.basic_consume(queue = self.callback_queue, on_message_callback = self.reply_callback, auto_ack = True)

                for event_type in self.listeners.keys():
                    self.consume_channel.queue_bind(
                        exchange = EVENTS_EXCHANGE_NAME,
                        queue = self.worker_queue_name,
                        routing_key = event_type)

                self.consume_channel.start_consuming()
            
            except pika.exceptions.ConnectionClosedByBroker:
                # Recovery from server-initiated connection closure, including
                # when the node is stopped cleanly
                _LOGGER.info("server-initiated connection closure, retrying...")
                continue
            # Do not recover on channel errors
            except pika.exceptions.AMQPChannelError:
                _LOGGER.exception("caught a channel error, stopping...")
                break
            # Recover on all other connection errors
            except pika.exceptions.AMQPConnectionError:
                if self.cancel_event.is_set():
                    break
                _LOGGER.exception("connection was closed, retrying...")
                continue

    def ensure_connection(self): 
        _LOGGER.info('client connecting to RabbitMQ ...')
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()

    def publish_internal_with_retry(self, exchange, routing_key, body, content_type, reply_to = None, correlation_id = None, retry_count = 3):
        retry = 0
        while retry < retry_count:
            try:
                self.ensure_connection()
                self.channel.basic_publish(
                    exchange = exchange,
                    routing_key = routing_key,
                    body = body,
                    properties = pika.BasicProperties (
                        content_type = content_type,
                        reply_to = reply_to,
                        correlation_id = correlation_id
                    )
                )
                return
            except Exception:
                retry = retry + 1
                _LOGGER.warn('failed to publish_event (reply_to: {properties.reply_to}, correlation_id: {properties.correlation_id}, retry {retry})')

        _LOGGER.error('all attempts failed to publish_event (reply_to: {properties.reply_to}, correlation_id: {properties.correlation_id})')

    def publish_event(self, event, reply_to = None, correlation_id = None):
        self.publish_internal_with_retry(
            exchange = EVENTS_EXCHANGE_NAME,
            routing_key = event['eventType'],
            body = json.dumps(event),
            content_type = 'application/json',
            reply_to = reply_to,
            correlation_id = correlation_id
        )

    def publish_reply(self, data, to, correlation_id, content_type = None):
        if content_type is None:
            content_type = 'application/json'
        self.publish_internal_with_retry(
            exchange = '',
            routing_key = to,
            body = json.dumps(data) if content_type == 'application/json' else data,
            content_type = content_type,
            correlation_id = correlation_id
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
                        result = target(event)

                        if isinstance(result, tuple):
                            result_data, content_type = result
                        else:
                            result_data, content_type = result, None

                        if properties.reply_to:
                            self.publish_reply(
                                data = result_data,
                                content_type = content_type,
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
            correlation_id = properties.correlation_id

            _LOGGER.debug(f"handling reply (correlation_id: {correlation_id})")	

            if correlation_id:	
                wait_event = self.wait_events.pop(correlation_id, None)
                if wait_event:
                    self.results[correlation_id] = (body, properties.content_type)
                    wait_event.set()

        except Exception:
            _LOGGER.exception('Error in reply_callback.')

    def listen(self, event_type: str, target: Callable[..., Any]) -> Callable[[], None]:
        _LOGGER.info(f"listening for {event_type}.")
        if event_type not in self.listeners:
            self.listeners[event_type] = []
            if self.consume_channel:
                self.consume_channel.queue_bind(
                    exchange = EVENTS_EXCHANGE_NAME,
                    queue = self.worker_queue_name,
                    routing_key = event_type)

        self.listeners[event_type].append(target)
    
    def register_service(self, service_name, target):
        #self.listeners[f'service.{name}'] = []
        self.listen(f'service.{service_name}', lambda x: target(x.get('serviceData')))

    def call_service(self, service_name, service_data, timeout = 300, parse_json = True):
        correlation_id = str(uuid.uuid4())
        _LOGGER.debug(f"call service '{service_name}' (correlation_id: {correlation_id}, timeout: {timeout}).")
        self.wait_events[correlation_id] = threading.Event()
        self.publish_event(
            event = {
                'eventType': f'service.{service_name}',
                'serviceData': service_data
            },
            reply_to = self.callback_queue,
            correlation_id = correlation_id
        )
        try:
            if self.wait_events[correlation_id].wait(timeout = timeout):
                body, content_type = self.results.pop(correlation_id)
                if content_type == 'application/json' and parse_json:
                    return json.loads(body)
                else:
                    return body, content_type
            else:
                _LOGGER.warn(f"call to '{service_name}' timed out.")
                return { 'error': 'timeout' }
        except:
            _LOGGER.exception(f"call service '{service_name}' failed.")
            return { 'error': 'failed' }
        finally:
            self.wait_events.pop(correlation_id, None)

    def start(self):
        if self.thread:
            _LOGGER.warning(f"cannot start event processing, since it has already been started.")
            return

        _LOGGER.info("event processing is starting ...")
        self.cancel_event = threading.Event()
        self.thread = threading.Thread(target=self.consume_internal)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        _LOGGER.info("event processing is stopping ...")
        
        if self.cancel_event:
            self.cancel_event.set()

        if self.consume_connection:
            try:
                self.consume_channel.stop_consuming()
                self.consume_connection.close()
            except pika.exceptions.AMQPConnectionError:
                pass

        if self.connection:
            try:
                self.channel.stop_consuming()
                self.connection.close()
            except pika.exceptions.AMQPConnectionError:
                pass

        if self.thread:
            self.thread.join(timeout=5) 

        self.cancel_event = None
        self.thread = None
