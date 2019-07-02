import os
import pika
import json

from vehicletracker.consts.events import *

EVENTS_EXCHANGE_NAME = 'vehicletracker-events'
LISTEN_EVENT_TYPES = [EVENT_LINK_TRAIN_COMPLETED,]

credentials = pika.PlainCredentials(
    username = os.environ['RABBITMQ_USERNAME'],
    password = os.environ['RABBITMQ_PASSWORD'])

parameters = pika.ConnectionParameters(
    host = os.environ['RABBITMQ_ADDRESS'],
    port = os.getenv('RABBITMQ_PORT', 5672),
    virtual_host='/',
    credentials=credentials)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()

train_event = {
    'eventType': EVENT_LINK_TRAIN_REQUEST,
    'linkRef': '1074:7051',
    'model': 'svr',
    'parameters': {
        'n': 21,
        'time': '2019-04-01'
    }
}

predict_event = {
    'eventType': EVENT_LINK_PREDICT_REQUEST,
    'linkRef': '1074:7051',
    'model': 'svr',
    'time': '2019-04-01'
}

event = predict_event

channel.basic_publish(EVENTS_EXCHANGE_NAME, routing_key = event['eventType'], body = json.dumps(event))
