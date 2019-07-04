import os
import logging
from logging.handlers import TimedRotatingFileHandler

from vehicletracker.azure.events import EventHubsQueue
from vehicletracker.data.client import PostgresClient

event_queue = EventHubsQueue(consumer_group = 'recorder')

_LOGGER = logging.getLogger('vehicletraker.recorder')

def main():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logHandler = TimedRotatingFileHandler("./logs/recorder.log", when="midnight")
    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logHandler.setFormatter(logFormatter)
    _LOGGER.addHandler(logHandler)
    _LOGGER.setLevel(logging.DEBUG)

    for l_ in ['vehicletracker.data.events', 'vehicletracker.azure.events']:
        l = logging.getLogger(l_)
        l.addHandler(logHandler)
        l.setLevel(logging.DEBUG)

    state_store = PostgresClient()    

    def log_event(event):
        uuid = event.get('correlationId')
        state_store.log_event(event, uuid)

    event_queue.listen('*', log_event)
    event_queue.start()

    input("Press Enter to exit...")
    os._exit(0)

    event_queue.stop()

if __name__ == '__main__':
    main()
