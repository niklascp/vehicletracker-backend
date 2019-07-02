import os
import pika
import json

from vehicletracker.data.client import PostgresClient
from vehicletracker.consts.events import *

EVENTS_EXCHANGE_NAME = 'vehicletracker-events'

def main():
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

    state_store = PostgresClient()    

    def callback(ch, method, properties, body):
        if properties.content_type == 'application/json':
            event = json.loads(body)
            uuid = properties.correlation_id
            state_store.log_event(event, uuid)

    queue_result = channel.queue_declare(queue = '', exclusive=True)

    channel.queue_bind(exchange=EVENTS_EXCHANGE_NAME, queue=queue_result.method.queue, routing_key='#')

    channel.basic_consume(
        queue=queue_result.method.queue,
        on_message_callback=callback,
        auto_ack=True)

    channel.start_consuming()

if __name__ == '__main__':
    main()
