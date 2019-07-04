import os
import logging
import threading

import uuid
import json

from typing import Any, Callable

from azure.eventhub import EventHubClient, Sender, EventData, Offset

_LOGGER = logging.getLogger(__name__)

class EventHubsQueue():  

    def __init__(self, consumer_group):

        OFFSET = Offset("@latest")
        PARTITION = "0"

        self.EVENT_HUB_ADDRESS = os.environ.get('EVENT_HUB_ADDRESS')

        if not self.EVENT_HUB_ADDRESS:
            raise ValueError("No EventHubs URL supplied.")

        # SAS policy and key are not required if they are encoded in the URL
        USER = os.environ.get('EVENT_HUB_SAS_POLICY')
        KEY = os.environ.get('EVENT_HUB_SAS_KEY')

        self.stop_event = threading.Event()
        self.client = EventHubClient(self.EVENT_HUB_ADDRESS, debug = False, username = USER, password = KEY)
        self.sender = self.client.add_sender(partition = PARTITION)
        self.receiver = self.client.add_receiver(consumer_group, PARTITION, prefetch = 5000, offset = OFFSET)

        self.wait_events = {}
        self.results = {}
        self.listeners = {}

        self.thread = threading.Thread(target=self.receive)

    def send(self, data):
        self.sender.send(EventData(json.dumps(data)))

    def receive(self):
        while not self.stop_event.is_set():
            batch = self.receiver.receive(timeout = 5)
            if batch:
                for msg in batch:
                    event = msg.body_as_json()
                    event_type = event['eventType']
                    correlation_id = event.get('correlationId')

                    target_list = self.listeners.get(event_type, []) + self.listeners.get('*', [])

                    _LOGGER.debug(f"handling {event_type} with {len(target_list)} targets (correlation_id: {correlation_id})")

                    for target in target_list:
                        # TODO: Wrap targets in exception logging and execute async.
                        try:
                            #task_runner.async_add_job(target, *args)
                            result_data = target(event)

                            if correlation_id and event_type != 'reply' and result_data:
                                self.publish_event({
                                        'eventType': 'reply',
                                        'data': result_data,
                                        'correlationId': correlation_id
                                    })
                        except Exception as e:
                            _LOGGER.exception('error in executing listerner')
                            self.publish_event({
                                    'eventType': 'error',
                                    'error': str(e)
                                })
                            if correlation_id and event_type != 'reply':
                                self.publish_event({
                                        'eventType': 'reply',
                                        'error': str(e),
                                        'correlationId': correlation_id
                                    })

    def reply_callback(self, event):
        correlation_id = event.get('correlationId')
        if correlation_id:
            wait_event = self.wait_events.pop(correlation_id, None)
            if wait_event:
                self.results[correlation_id] = event.get('data')
                wait_event.set()
        else:
            _LOGGER.warning(f"reply event is missing correlation_id.")

    def publish_event(self, event):
        _LOGGER.debug(f"publishing event ({event['eventType']}, {event.get('correlationId')}).")
        self.send(event)

    def listen(self, event_type: str, target: Callable[..., Any]) -> Callable[[], None]:
        _LOGGER.info(f"listening for {event_type}.")
        if event_type not in self.listeners:
            self.listeners[event_type] = []

        self.listeners[event_type].append(target)

    def register_service(self, service_name, target):
        #self.listeners[f'service.{name}'] = []
        self.listen(f'service.{service_name}', lambda x: target(x.get('serviceData')))

    def call_service(self, service_name, service_data, timeout = 1):
        correlation_id = str(uuid.uuid4())
        self.wait_events[correlation_id] = threading.Event()
        self.publish_event({
                'eventType': f'service.{service_name}',
                'correlationId': correlation_id,
                'serviceData': service_data
            })
        try:
            if self.wait_events[correlation_id].wait(timeout = timeout):
                return self.results.pop(correlation_id)
        except:
            raise
        finally:
            self.wait_events.pop(correlation_id, None)

    def start(self):
        self.listen('reply', self.reply_callback)

        _LOGGER.info(f"connecting to event hub at '{self.EVENT_HUB_ADDRESS}'")        
        self.client.run()
        self.thread.start()

    def stop(self):        
        self.stop_event.set()
        self.thread.join(5)
        self.client.stop()
