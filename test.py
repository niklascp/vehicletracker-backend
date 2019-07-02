import os
import signal

from vehicletracker.data.events import EventQueue

event_queue = EventQueue('test')
event_queue.start()

result = event_queue.call_service(
    service_name = 'link.predict',
    service_data = {
        'linkRef': '1074:7051',
        'model': 'svr',
        'time': '2019-04-01'
})
print(result)
    
input("Press Enter to exit...")
os._exit(0)