"""
Core components of Vehicle Tracker.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import enum
import logging
import threading
import json
import uuid

from typing import (
    Optional,
    Any,
    Callable,
    Coroutine,
    List,
    Dict,
    TypeVar,
)

import aio_pika
from aio_pika.exchange import ExchangeType

T = TypeVar("T")
CALLABLE_T = TypeVar("CALLABLE_T", bound=Callable)
CALLBACK_TYPE = Callable[[], None]

_LOGGER = logging.getLogger(__name__)

def callback(func: CALLABLE_T) -> CALLABLE_T:
    """Annotation to mark method as safe to call from within the event loop."""
    setattr(func, "_hass_callback", True)
    return func

def is_callback(func: Callable[..., Any]) -> bool:
    """Check if function is safe to be called in the event loop."""
    return getattr(func, "_hass_callback", False) is True

@callback
def async_loop_exception_handler(_: Any, context: Dict) -> None:
    """Handle all exception inside the core loop."""
    kwargs = {}
    exception = context.get("exception")
    if exception:
        kwargs["exc_info"] = (type(exception), exception, exception.__traceback__)

    _LOGGER.error(  # type: ignore
        "Error doing job: %s", context["message"], **kwargs
    )

class NodeState(enum.Enum):
    """Represent the current state of the node."""

    not_running = "NOT_RUNNING"
    starting = "STARTING"
    running = "RUNNING"
    stopping = "STOPPING"

    def __str__(self) -> str:
        """Return the state."""
        return self.value  # type: ignore

class VehicleTrackerNode:
    """Root node object of the Vehicle Tracker system."""

    def __init__(self, config : Dict[str, Any], loop: Optional[asyncio.events.AbstractEventLoop] = None) -> None:
        """Initialize new Vehicle Tracker node object."""
        self.name = config.get('node', {}).get('name', None) or 'default'
        self.loop: asyncio.events.AbstractEventLoop = (loop or asyncio.get_event_loop())

        executor_opts: Dict[str, Any] = {
            "max_workers": None,
            "thread_name_prefix": "SyncWorker",
        }

        self.executor = ThreadPoolExecutor(**executor_opts)
        self.loop.set_default_executor(self.executor)
        self.loop.set_exception_handler(async_loop_exception_handler)
        self._pending_tasks: list = []
        self._track_task = True
        self.events = EventBus(self)
        self.services = ServiceBus(self)
        # This is a dictionary that any component can store any data on.
        self.data: dict = {}
        self.state = NodeState.not_running
        self.exit_code = 0
        # If not None, use to signal end-of-loop
        self._stopped: Optional[asyncio.Event] = None

    async def async_run(self, *, attach_signals: bool = True) -> int:
        """Vehicle Tracker main entry point.
        Start the Vehicle Tracker Node and block until stopped.
        This method is a coroutine.
        """
        if self.state != NodeState.not_running:
            raise RuntimeError("Vehicle Tracker is already running")

        # _async_stop will set this instead of stopping the loop
        self._stopped = asyncio.Event()

        await self.async_start()
        if attach_signals:
            from vehicletracker.helpers.signal import async_register_signal_handling
            async_register_signal_handling(self)

        #import time
        #while True:
        #    time.sleep(1)
        #    print("a")

        await asyncio.sleep(0)
        print( self.state)
        await self._stopped.wait()
        print("Here")
        return self.exit_code

    async def async_start(self) -> None:
        """Finalize startup from inside the event loop.
        This method is a coroutine.
        """
        _LOGGER.info("Starting Vehicle Tracker Node")
        self.state = NodeState.starting

        setattr(self.loop, "_thread_ident", threading.get_ident())
        #self.bus.async_fire(EVENT_HOMEASSISTANT_START)


        await self.async_block_till_done()
        #try:
        #    # Only block for EVENT_HOMEASSISTANT_START listener
        #    self.async_stop_track_tasks()
        #    with timeout(TIMEOUT_EVENT_START):
        #        await self.async_block_till_done()
        #except asyncio.TimeoutError:
        #    _LOGGER.warning(
        #        "Something is blocking Home Assistant from wrapping up the "
        #        "start up phase. We're going to continue anyway. Please "
        #        "report the following info at http://bit.ly/2ogP58T : %s",
        #        ", ".join(self.config.components),
        #    )

        self.state = NodeState.running
        #_async_create_timer(self)

    def stop(self) -> None:
        """Stop Vehicle Tracker Node and shuts down all threads."""
        if self.state == NodeState.not_running:  # just ignore
            return
        asyncio.ensure_future(self.async_stop(), loop = self.loop)

    async def async_stop(self, exit_code: int = 0, *, force: bool = False) -> None:
        """Stop Vehicle Tracker Node and shuts down all threads.
        The "force" flag commands async_stop to proceed regardless of
        Home Assistan't current state. You should not set this flag
        unless you're testing.
        This method is a coroutine.
        """
        if not force:
            # Some tests require async_stop to run,
            # regardless of the state of the loop.
            if self.state == NodeState.not_running:  # just ignore
                return
            if self.state == NodeState.stopping:
                _LOGGER.info("async_stop called twice: ignored")
                return
            if self.state == NodeState.starting:
                # This may not work
                _LOGGER.warning("async_stop called before startup is complete")

        # stage 1
        self.state = NodeState.stopping
        #self.async_track_tasks()
        #self.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
        await self.async_block_till_done()

        # stage 2
        self.state = NodeState.not_running
        #self.bus.async_fire(EVENT_HOMEASSISTANT_CLOSE)
        await self.async_block_till_done()
        self.executor.shutdown()

        self.exit_code = exit_code
        print("Here 2")
        if self._stopped is not None:
            self._stopped.set()
        else:
            self.loop.stop()

    def block_till_done(self) -> None:
        """Block till all pending work is done."""
        asyncio.ensure_future(self.async_block_till_done(), loop = self.loop).result()

    async def async_block_till_done(self) -> None:
        """Block till all pending work is done."""
        # To flush out any call_soon_threadsafe
        await asyncio.sleep(0)

        while self._pending_tasks:
            pending = [task for task in self._pending_tasks if not task.done()]
            self._pending_tasks.clear()
            if pending:
                await asyncio.wait(pending)
            else:
                await asyncio.sleep(0)

    def add_job(self, target: Callable[..., Any], *args: Any) -> None:
        """Add job to the executor pool.
        target: target to call.
        args: parameters for method to call.
        """
        if target is None:
            raise ValueError("Don't call add_job with None")
        self.loop.call_soon_threadsafe(self.async_add_job, target, *args)

    @callback
    def async_add_job(
        self, target: Callable[..., Any], *args: Any
    ) -> Optional[asyncio.Future]:
        """Add a job from within the event loop.
        This method must be run in the event loop.
        target: target to call.
        args: parameters for method to call.
        """
        task = None

        # Check for partials to properly determine if coroutine function
        check_target = target
        while isinstance(check_target, functools.partial):
            check_target = check_target.func

        if asyncio.iscoroutine(check_target):
            task = self.loop.create_task(target)  # type: ignore
        elif is_callback(check_target):
            self.loop.call_soon(target, *args)
        elif asyncio.iscoroutinefunction(check_target):
            task = self.loop.create_task(target(*args))
        else:
            task = self.loop.run_in_executor(  # type: ignore
                None, target, *args
            )

        # If a task is scheduled
        if self._track_task and task is not None:
            self._pending_tasks.append(task)

        return task

    @callback
    def async_create_task(self, target: Coroutine) -> asyncio.tasks.Task:
        """Create a task from within the eventloop.
        This method must be run in the event loop.
        target: target to call.
        """
        task: asyncio.tasks.Task = self.loop.create_task(target)

        if self._track_task:
            self._pending_tasks.append(task)

        return task

EVENTS_EXCHANGE_NAME = 'vehicletracker-events'


class EventBus:
    """Allow publication of and listening for events over RabbitMQ."""

    def __init__(self, node: VehicleTrackerNode) -> None:
        """Initialize a new event bus."""
        self._connection : aio_pika.Connection = None
        self._channel : aio_pika.Channel = None
        self._domain_queues: Dict[str, aio_pika.Queue] = {}
        self._listeners: Dict[str, Dict[str, List[Callable]]] = {} # map from domain -> event_type -> targets
        self._node = node
        self._future = asyncio.ensure_future(self.async_connect(), loop = self._node.loop)

    async def async_connect(self):
        """Initialize connection to  new event bus."""
        _LOGGER.info("Connecting to RabbitMQ ...")
        self._connection = await aio_pika.connect_robust("amqp://guest:guest@127.0.0.1/", loop = self._node.loop)
        # Creating channel
        self._channel = await self._connection.channel()    # type: aio_pika.Channel
        self._event_exchange = await self._channel.declare_exchange(EVENTS_EXCHANGE_NAME, ExchangeType.TOPIC)
        return True

    async def _async_listen_queue(self, domain: str, queue : aio_pika.Queue) -> None:
        """Loop for listening on a specific queue""" 
        async with queue.iterator() as queue_iter:
            _LOGGER.info("Proccessing of queue '%s' is starting.", queue.name)
            # Cancel consuming after __aexit__
            async for message in queue_iter:
                async with message.process():
                    event_type = message.headers['event_type']
                    domain_listeners = self._listeners.get(domain, {})
                    print(event_type, domain, len(domain_listeners))
                    event = json.loads(message.body)
                    for target in domain_listeners.get(event_type, []):
                        self._node.async_add_job(target, event)
                    for target in domain_listeners.get('*', []):
                        self._node.async_add_job(target, event)

    async def _async_remove_listener(self, domain : str, event_type : str, target : Callable):
        """Remove a listener of a specific event_type.
        This method must be run in the event loop.
        """
        try:
            self._listeners[domain][event_type].remove(target)

            # delete event_type list if empty
            if not self._listeners[domain][event_type]:
                self._listeners[domain].pop(event_type)

            # delete domain if empty
            if not self._listeners[domain]:
                self._listeners.pop(domain)
        except (KeyError, ValueError):
            # KeyError is key event_type listener did not exist
            # ValueError if listener did not exist within event_type
            _LOGGER.warning("Unable to remove unknown listener %s", target)

    def listen(
        self,
        event_type : str,
        target: Callable) -> Callable[[], None]:
        """
        Listen for the given event_type on this node
        """
        return self.listen_domain(
            None,
            event_type,
            target
        )

    async def async_listen(
        self,
        event_type : str,
        target: Callable) -> Callable[[], None]:
        """
        Listen for the given event_type on this node
        This method must be run in the event loop.
        """
        return await self.async_listen_domain(
            None,
            event_type,
            target
        )

    def listen_domain(
        self,
        domain : str,
        event_type : str,
        target: Callable) -> Callable[[], None]:
        """
        Listen for the given event_type on the domain
        """
        return asyncio.run_coroutine_threadsafe(
            self.async_listen_domain(domain, event_type, target), 
            loop = self._node.loop).result()

    async def async_listen_domain(
        self,
        domain : str,
        event_type : str,
        target: Callable[..., Any]) -> Callable[[], None]:
        """Listen for the given event_type on the domain"""

        if domain:
            _LOGGER.info("Listening for %s (domain).", event_type)
        else:
            _LOGGER.info("Listening for %s (node).", event_type)
            domain = 'node-' + self._node.name

        if not domain in self._domain_queues:
            await self._future
            # Declaring the domain queue
            queue = await self._channel.declare_queue(
                domain,
                auto_delete=True
            ) # type: aio_pika.Queue
            # Start consume from the domain queue
            self._domain_queues[domain] = queue
            self._listeners[domain] = {}
            self._node.loop.create_task(self._async_listen_queue(domain, queue))

        if event_type in self._listeners[domain]:
            self._listeners[domain][event_type].append(target)
        else:
            await self._domain_queues[domain].bind(self._event_exchange, '#' if event_type == '*' else event_type)
            self._listeners[domain][event_type] = [target]

        def remove_listener() -> None:
            """Remove the listener."""
            self._async_remove_listener(domain, event_type, target)

        return remove_listener

    async def async_publish(self, event):
        """Publish an event."""
        await self._event_exchange.publish(
            aio_pika.Message(
                bytes(json.dumps(event), 'utf-8'),
                content_type='application/json',
                headers={
                    'event_type': event['eventType']
                }
            ),
            event['eventType']
        )

    async def async_reply(self, event, to_node):
        """Publish a reply."""
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                bytes(json.dumps(event), 'utf-8'),
                content_type='application/json',
                headers={
                    'event_type': event['eventType']
                }
            ),
            to_node
        )

class ServiceBus:
    """Offer the services over the eventbus."""

    def __init__(self, node: VehicleTrackerNode) -> None:
        """Initialize a service registry."""
        self._node = node
        self._wait_events : Dict[str, asyncio.Event] = {}
        self._results : Dict[str, Any] = {}
        asyncio.ensure_future(self._node.events.async_listen('reply', self._async_handle_reply))

    @callback
    def _async_handle_reply(self, event):
        correlation_id = event['correlationId']
        result = event['result']
        self._results[correlation_id] = result
        self._wait_events[correlation_id].set()

    def register(
        self,
        domain: str,
        service: str,
        service_func: Callable
    ) -> None:
        """
        Register a service.
        """
        asyncio.run_coroutine_threadsafe(
            self.async_register(domain, service, service_func), 
            loop = self._node.loop).result()

    @callback
    async def async_register(
        self,
        domain: str,
        service: str,
        service_func: Callable
    ) -> None:
        """
        Register a service.
        This method must be run in the event loop.
        """
        domain = domain.lower()
        service = service.lower()

        async def service_wrapper(event : Dict[str, Any]):
            service_data = event['serviceData']
            correlation_id = event['correlationId']
            timeout = 30
            reply_to = event['replyTo']

            _LOGGER.info("Handling service request for '%s' (correlation_id: %s, timeout: %s, reply_to = %s).", 
                service, correlation_id, timeout, reply_to)

            try:
                result = await self._node.async_add_job(service_func, service_data)
                await self._node.events.async_reply({
                        'eventType': 'reply',
                        'result': result,
                        'correlationId': correlation_id,
                    }, to_node = reply_to)
            except Exception as ex: # pylint: disable=broad-except
                _LOGGER.exception("Error in executing service '%s' (correlation_id: %s, timeout: %s, reply_to = %s)",
                    service, correlation_id, timeout, reply_to)
                await self._node.events.async_reply({
                        'eventType': 'reply',
                        'result': {
                            'error': str(ex)
                        },
                        'correlationId': correlation_id,
                    }, to_node = reply_to)

        await asyncio.ensure_future(
            self._node.events.async_listen_domain(domain, service, service_wrapper),
            loop = self._node.loop)

    def call(
        self,
        service: str,
        service_data: Optional[Dict] = None,
        timeout: int = 30, 
        parse_json: bool = True
    ) -> Any:
        """
        Call a service and return result.
        """
        return asyncio.run_coroutine_threadsafe(  # type: ignore
            self.async_call(service, service_data, timeout, parse_json),
            loop = self._node.loop,
        ).result()

    async def async_call(
        self,
        service: str,
        service_data: Optional[Dict] = None,
        timeout: int = 30, 
        parse_json: bool = True
    ) -> Optional[bool]:
        """
        Call a service.

        This method is a coroutine.
        """
        service = service.lower()
        service_data = service_data or {}

        correlation_id = str(uuid.uuid4())
        self._wait_events[correlation_id] = asyncio.Event()
        _LOGGER.info("Call service '%s' (correlation_id: %s, timeout: %s).", service, correlation_id, timeout)

        task = self._node.events.async_publish({
            'eventType': service,
            'serviceData': service_data,
            'replyTo': 'node-' + self._node.name,
            'correlationId': correlation_id,
        })

        await self._node.async_create_task(task)

        try:
            await asyncio.wait_for(self._wait_events[correlation_id].wait(), timeout)
            return self._results.pop(correlation_id)
        except asyncio.TimeoutError:
            _LOGGER.warning("call to '%s' timed out.", service)
            return { 'error': 'timeout' }
        except Exception: # pylint: disable=broad-except
            _LOGGER.exception("call service '%s' failed.", service)
            return { 'error': 'failed' }
        finally:
            self._wait_events.pop(correlation_id, None)

async def async_setup_components(node : VehicleTrackerNode, config : Dict[str, Any]) -> None:
    """Set up all the components.
    
    This method is a coroutine.
    """

    core_components = ['node']

    # Set up core.
    components = list(config.keys())

    _LOGGER.info("Setting up %s", components)

    if not all(
        await asyncio.gather(
            *(
                async_setup_component(node, domain, config)
                for domain in components
                if domain not in core_components
            )
        )
    ):
        _LOGGER.error(
            "Failed to initialize all components."
        )
        return

async def async_setup_component(
    node : VehicleTrackerNode,
    domain: str,
    config : Dict[str, Any]) -> None:
    """Set up a single component.
        
    This method is a coroutine.
    """
    import importlib
   
    try:
        module_name = f"vehicletracker.components.{domain}"
        component = importlib.import_module(module_name)

        result = await component.async_setup(  # type: ignore
                    node, config
                )

        return result
    except Exception:
        _LOGGER.exception("Failed to setup component '%s'.", domain)
        return False
