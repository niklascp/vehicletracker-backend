import os
import uuid
import socket
import threading
import logging

import json

from typing import Any, Callable

from azure.servicebus import ServiceBusClient, QueueClient, Message
from azure.common import AzureConflictHttpError

from vehicletracker.common.exceptions import ApplicationError

_LOGGER = logging.getLogger(__name__)

class ServiceQueue():

    def __init__(self, domain, worker = False):
        self.topic = 'events'
        self.domain = domain
        self.worker = worker
        
        connection_string = os.getenv('SERVICE_BUS_CONNECTION_STRING')
        self.client_queue_name = domain + '-client-' + socket.gethostname()

        self.sb_client = ServiceBusClient.from_connection_string(connection_string)

        self.stop_event = threading.Event()
        self.wait_events = {}
        self.results = {}
        self.listeners = {}

        if self.worker:
            self.receive_worker_thread = threading.Thread(target=self.receive_worker)

        self.receive_client_thread = threading.Thread(target=self.receive_client)

    def send(self, to, event, correlation_id = None, reply_to = None):
        _LOGGER.debug(f"publishing event '{event['eventType']}' to '{to.name}' (correlation id: {correlation_id}).")
        msg = Message(json.dumps(event))
        msg.properties.content_type = 'application/json'
        if correlation_id:
            msg.properties.correlation_id = str(correlation_id)
        if reply_to:
            msg.properties.reply_to = str(reply_to)

        to.send([msg])

    def receive_worker(self):
        while not self.stop_event.is_set():
            with self.worker_subscription.get_receiver() as batch:
                while not self.stop_event.is_set():
                    for msg in batch.fetch_next(timeout=5):
                        try:
                            event = json.loads(str(msg))
                            event_type = event['eventType']
                            if msg.properties.correlation_id:
                                correlation_id = msg.properties.correlation_id.decode('utf-8') 
                            if msg.properties.reply_to:
                                reply_to = msg.properties.reply_to.decode('utf-8') 

                            if reply_to:
                                reply_to_queue = self.sb_client.get_queue(reply_to)

                            target_list = self.listeners.get(event_type, []) + self.listeners.get('*', [])

                            _LOGGER.debug(f"handling {event_type} with {len(target_list)} targets (correlation_id: {correlation_id})")

                            for target in target_list:
                                # TODO: Wrap targets in exception logging and execute async.
                                try:
                                    #task_runner.async_add_job(target, *args)
                                    result_data = target(event)

                                    if reply_to:
                                        self.send(
                                            to = reply_to_queue, 
                                            event = {
                                                'eventType': 'reply',
                                                'data': result_data
                                            },
                                            correlation_id = correlation_id)
                                except Exception as e:
                                    _LOGGER.exception('error in executing listerner')
                                    self.publish_event({
                                            'eventType': 'error',
                                            'error': str(e)
                                        })
                                    if reply_to:
                                        self.send(
                                            to = reply_to_queue, 
                                            event = {
                                                'eventType': 'reply',
                                                'data': result_data
                                            },
                                            correlation_id = correlation_id)
                            msg.complete()
                        except Exception:
                            _LOGGER.exception('error in receive_worker message loop')
                        #finally:
                        #    msg.complete()

    def receive_client(self):
        while not self.stop_event.is_set():
            with self.client_queue.get_receiver as batch:
                while not self.stop_event.is_set():
                    for msg in batch.fetch_next(timeout=5):
                        try:
                            event = json.loads(str(msg))
                            event_type = event['eventType']
                            correlation_id = msg.properties.correlation_id.decode('utf-8') 
                            
                            _LOGGER.debug(f"handling {event_type} (correlation_id: {correlation_id})")

                            if correlation_id:
                                wait_event = self.wait_events.pop(correlation_id, None)
                                if wait_event:
                                    self.results[correlation_id] = event.get('data')
                                    wait_event.set()
                            
                            msg.complete()
                        except Exception:
                            _LOGGER.exception('error in receive_client message loop')
                        #finally:
                        #    msg.complete()

    def listen(self, event_type: str, target: Callable[..., Any]) -> Callable[[], None]:
        _LOGGER.info(f"listening for {event_type}.")
        if event_type not in self.listeners:
            self.listeners[event_type] = []

        self.listeners[event_type].append(target)

    def publish_event(self, event, correlation_id = None, reply_to = None):
        self.send(self.events_topic, event, correlation_id, reply_to)

    def register_service(self, service_name, target):
        if not self.worker:
            raise ApplicationError("Cannot register services on a client-only ServiceQueue.")
        self.listen(f'service.{service_name}', lambda x: target(x.get('serviceData')))

    def call_service(self, service_name, service_data, timeout = 1):
        correlation_id = str(uuid.uuid4())
        self.wait_events[correlation_id] = threading.Event()
        self.publish_event({
                'eventType': f'service.{service_name}',
                'serviceData': service_data
            },
            correlation_id = correlation_id,
            reply_to = self.client_queue_name)
        try:
            if self.wait_events[correlation_id].wait(timeout = timeout):
                return self.results.pop(correlation_id)
            else:
                _LOGGER.info(f"call to {service_name} timed out.")
        except:
            raise
        finally:
            self.wait_events.pop(correlation_id, None)

    def start(self):
        _LOGGER.info(f"service bus processing is starting")

        try:
            self.sb_client.create_topic(self.topic)
        except AzureConflictHttpError:
            _LOGGER.debug(f"topic '{self.topic}' already exists.")
            pass

        try:
            if self.worker:
                self.sb_client.create_subscription(self.topic, self.domain)
        except AzureConflictHttpError:
            _LOGGER.debug(f"subscription '{self.domain}' already exists.")
            pass

        try:
            self.sb_client.create_queue(self.client_queue_name)
        except AzureConflictHttpError:
            _LOGGER.debug(f"queue '{self.client_queue_name}' already exists.")
            pass

        # Create the QueueClient 
        self.events_topic = self.sb_client.get_topic(self.topic)

        self.client_queue = self.sb_client.get_queue(self.client_queue_name)      
        self.receive_client_thread.start()

        if self.worker:
            self.worker_subscription = self.sb_client.get_subscription(self.topic, self.domain)
            self.receive_worker_thread.start()    

    def stop(self):
        self.stop_event.set()
        if self.worker:
            self.receive_worker_thread.join(5)
        self.receive_client_thread.join(5)
