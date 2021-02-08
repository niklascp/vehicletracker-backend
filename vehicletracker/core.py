"""
Core components of Vehicle Tracker.
"""
import asyncio
import datetime as dt
import enum
import functools
import json
import logging
import os
import threading
import uuid
from time import monotonic
from typing import (Any, Awaitable, Callable, Coroutine, Dict, Iterable, List,
                    Optional, TypeVar)

import aio_pika
import pytz
from aio_pika.exchange import ExchangeType
from async_timeout import timeout

from vehicletracker.const import (ATTR_EVENT_TYPE, ATTR_NODE_NAME, ATTR_NOW,
                                  EVENT_NODE_START, EVENT_NODE_STOP,
                                  EVENT_REPLY, EVENT_TIME_CHANGED,
                                  TIMEOUT_EVENT_START, TIMEOUT_EVENT_STOP)
from vehicletracker.helpers.json import DateTimeEncoder

T = TypeVar("T")
CALLABLE_T = TypeVar("CALLABLE_T", bound=Callable)
CALLBACK_TYPE = Callable[[], None]

# How long to wait to log tasks that are blocking
BLOCK_LOG_TIMEOUT = 60

# RabbitMQ Exchange for events
EVENTS_EXCHANGE_NAME = 'vehicletracker-events'

_LOGGER = logging.getLogger(__name__)

def callback(func: CALLABLE_T) -> CALLABLE_T:
    """Annotation to mark method as safe to call from within the event loop."""
    setattr(func, "_hass_callback", True)
    return func

def is_callback(func: Callable[..., Any]) -> bool:
    """Check if function is safe to be called in the event loop."""
    return getattr(func, "_hass_callback", False) is True

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

    def __init__(self, config : Dict[str, Any]) -> None:
        """Initialize new Vehicle Tracker node object."""
        self.name = config.get('node', {}).get('name', None) or 'default'
        self.loop = asyncio.get_running_loop()
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

        await self._stopped.wait()
        return self.exit_code

    async def async_start(self) -> None:
        """Finalize startup from inside the event loop.
        This method is a coroutine.
        """
        _LOGGER.info("Starting Vehicle Tracker Node")
        self.state = NodeState.starting

        setattr(self.loop, "_thread_ident", threading.get_ident())
        await self.events.async_publish(EVENT_NODE_START, {
            ATTR_NODE_NAME: self.name
        })

        await self.async_block_till_done()
        try:
            # Only block for EVENT_NODE_START listener
            self.async_stop_track_tasks()
            with timeout(TIMEOUT_EVENT_START):
                await self.async_block_till_done()
            
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Something is blocking Vehicle Tracker from wrapping up the "
                "start up phase. We're going to continue anyway."
            )

        self.state = NodeState.running
        await _async_create_timer(self)

    def stop(self) -> None:
        """Stop Vehicle Tracker Node and shuts down all threads."""
        if self.state == NodeState.not_running:  # just ignore
            return
        asyncio.ensure_future(self.async_stop(), loop = self.loop)

    async def async_stop(self, exit_code: int = 0, *, force: bool = False) -> None:
        """Stop Vehicle Tracker Node and shuts down all threads.
        The "force" flag commands async_stop to proceed regardless of
        the nodes current state. You should not set this flag
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
        self.async_track_tasks()
        await self.events.async_publish(EVENT_NODE_STOP, {
            ATTR_NODE_NAME: self.name
        })
        try:
            async with timeout(120):
                await self.async_block_till_done()
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Timed out waiting for shutdown stage 1 to complete, the shutdown will continue"
            )

        # stage 2
        self.state = NodeState.not_running
        await self.events.async_close()

        if hasattr(self.loop, "shutdown_default_executor"):
            await self.loop.shutdown_default_executor()  # type: ignore

        self.exit_code = exit_code

        if self._stopped is not None:
            self._stopped.set()
        else:
            self.loop.stop()

    def block_till_done(self) -> None:
        """Block until all pending work is done."""
        asyncio.run_coroutine_threadsafe(
            self.async_block_till_done(), self.loop
        ).result()

    async def async_block_till_done(self) -> None:
        """Block until all pending work is done."""
        # To flush out any call_soon_threadsafe
        await asyncio.sleep(0)
        start_time: Optional[float] = None

        while self._pending_tasks:
            pending = [task for task in self._pending_tasks if not task.done()]
            self._pending_tasks.clear()
            if pending:
                await self._await_and_log_pending(pending)

                if start_time is None:
                    # Avoid calling monotonic() until we know
                    # we may need to start logging blocked tasks.
                    start_time = 0
                elif start_time == 0:
                    # If we have waited twice then we set the start
                    # time
                    start_time = monotonic()
                elif monotonic() - start_time > BLOCK_LOG_TIMEOUT:
                    # We have waited at least three loops and new tasks
                    # continue to block. At this point we start
                    # logging all waiting tasks.
                    for task in pending:
                        _LOGGER.debug("Waiting for task: %s", task)
            else:
                await asyncio.sleep(0)

    async def _await_and_log_pending(self, pending: Iterable[Awaitable[Any]]) -> None:
        """Await and log tasks that take a long time."""
        wait_time = 0
        while pending:
            _, pending = await asyncio.wait(pending, timeout=BLOCK_LOG_TIMEOUT)
            if not pending:
                return
            wait_time += BLOCK_LOG_TIMEOUT
            for task in pending:
                _LOGGER.debug("Waited %s seconds for task: %s", wait_time, task)

    @callback
    def async_run_job(self, target: Callable[..., None], *args: Any) -> None:
        """Run a job from within the event loop.
        This method must be run in the event loop.
        target: target to call.
        args: parameters for method to call.
        """
        if (
            not asyncio.iscoroutine(target)
            and not asyncio.iscoroutinefunction(target)
            and is_callback(target)
        ):
            target(*args)
        else:
            self.async_add_job(target, *args)

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

    @callback
    def async_track_tasks(self) -> None:
        """Track tasks so you can wait for all tasks to be done."""
        self._track_task = True

    @callback
    def async_stop_track_tasks(self) -> None:
        """Stop track tasks so you can't wait for all tasks to be done."""
        self._track_task = False

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
        self._json_encoder = DateTimeEncoder()

    async def async_connect(self):
        """Initialize connection to  new event bus."""
        _LOGGER.info("Connecting to RabbitMQ ...")
        self._connection = await aio_pika.connect_robust(
            os.getenv('RABBITMQ_URL', 'amqp://guest:guest@127.0.0.1/'),
            loop = self._node.loop)
        # Creating channel
        self._channel = await self._connection.channel()    # type: aio_pika.Channel
        self._event_exchange = await self._channel.declare_exchange(EVENTS_EXCHANGE_NAME, ExchangeType.TOPIC)
        return True

    async def async_close(self):
        if self._connection:
            _LOGGER.info("Closing connection to RabbitMQ ...")
            await self._connection.close()

    async def _async_listen_queue(self, domain: str, queue : aio_pika.Queue) -> None:
        """Loop for listening on a specific queue""" 
        _LOGGER.info("Proccessing of queue '%s' is starting.", queue.name)
        try:
            async for message in queue:
                async with message.process():
                    event_type = message.headers['event_type']
                    event_data = json.loads(message.body)
                    self.publish_local(event_type, event_data, domain)
        except:
            _LOGGER.info("Proccessing of queue '%s' has stopped.", queue.name)

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

    async def async_listen_domain(
        self,
        domain : str,
        event_type : str,
        target: Callable[..., Any]) -> Callable[[], None]:
        """Listen for the given event_type on the domain"""

        async def _consume(message : aio_pika.Message):
            event_type = message.headers['event_type']
            event_data = json.loads(message.body)
            self.publish_local(event_type, event_data, domain)

        if domain:
            _LOGGER.info("Listening for %s (domain: %s).", event_type, domain)
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
            self._node.async_create_task(queue.consume(_consume, no_ack=True))
            self._listeners[domain] = {}
            #self._node.loop.create_task(self._async_listen_queue(domain, queue))

        if event_type in self._listeners[domain]:
            self._listeners[domain][event_type].append(target)
        else:
            await self._domain_queues[domain].bind(self._event_exchange, '#' if event_type == '*' else event_type + '.#')
            self._listeners[domain][event_type] = [target]

        def remove_listener() -> None:
            """Remove the listener."""
            self._async_remove_listener(domain, event_type, target)

        return remove_listener

    def publish_local(self, event_type, event_data : Dict[str, Any], domain : str = None) -> None:
        if domain:            
            _LOGGER.info("Publishing local event '%s' for domain '%s'", event_type, domain)
        else:
            if event_type != EVENT_TIME_CHANGED:
                _LOGGER.info("Publishing local event '%s' for node", event_type)
            domain = 'node-' + self._node.name
        
        domain_listeners = self._listeners.get(domain, {})
        for target in domain_listeners.get(event_type, []):
            self._node.async_add_job(target, event_type, event_data)
        for target in domain_listeners.get('*', []):
            self._node.async_add_job(target, event_type, event_data)

    def publish(self, event_type : str, event_data : Dict[str, Any]) -> None:
        """Publish an event."""
        return asyncio.run_coroutine_threadsafe(
            self.async_publish(event_type, event_data), 
            loop = self._node.loop).result()

    async def async_publish(self, event_type : str, event_data : Dict[str, Any]):
        """Publish an event."""

        _LOGGER.info("Publishing global event '%s'", event_type)

        await self._future
        await self._event_exchange.publish(
            aio_pika.Message(
                bytes(self._json_encoder.encode(event_data), 'utf-8'),
                content_type='application/json',
                headers={
                    'event_type': event_type
                }
            ),
            event_type
        )

    async def async_reply(self, event_type : str, event_data : Dict[str, Any], to_node):
        """Publish a reply."""
        await self._future
        await self._channel.default_exchange.publish(
            aio_pika.Message(
                bytes(self._json_encoder.encode(event_data), 'utf-8'),
                content_type='application/json',
                headers={
                    'event_type': event_type
                }
            ),
            to_node
        )

class ServiceBus:
    """Offer Services over the Event Bus."""

    def __init__(self, node: VehicleTrackerNode) -> None:
        """Initialize a service registry."""
        self._node = node
        self._wait_events : Dict[str, asyncio.Event] = {}
        self._results : Dict[str, Any] = {}
        asyncio.ensure_future(self._node.events.async_listen('reply', self._async_handle_reply))

    @callback
    def _async_handle_reply(self, event_type, event_data):
        correlation_id = event_data['correlationId']
        result = event_data['result']
        if correlation_id in self._wait_events:
            try:
                self._results[correlation_id] = result
                self._wait_events[correlation_id].set()
            except KeyError:
                self._results.pop(correlation_id, None)
                pass
        else:
            logging.debug('Recived late reply (correlation_id: %s)', correlation_id)

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

        async def service_wrapper(event_type : str, event_data : Dict[str, Any]):
            service_data = event_data['serviceData']
            correlation_id = event_data['correlationId']
            timeout = 30
            reply_to = event_data['replyTo']

            _LOGGER.info("Handling service request for '%s' (correlation_id: %s, timeout: %s, reply_to = %s).", 
                service, correlation_id, timeout, reply_to)

            try:
                result = await self._node.async_add_job(service_func, service_data)
                await self._node.events.async_reply(EVENT_REPLY, {
                        'result': result,
                        'correlationId': correlation_id,
                    }, to_node = reply_to)
            except Exception as ex: # pylint: disable=broad-except
                _LOGGER.exception("Error in executing service '%s' (correlation_id: %s, timeout: %s, reply_to = %s)",
                    service, correlation_id, timeout, reply_to)
                await self._node.events.async_reply(EVENT_REPLY, {
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

        task = self._node.events.async_publish(service, {
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

async def _async_create_timer(node: VehicleTrackerNode) -> None:
    """Create a timer that will start on NODE_START."""
    handle = None

    def schedule_tick(now) -> None:
        """Schedule a timer tick when the next second rolls around."""
        nonlocal handle
        
        slp_seconds = 1 - (now.microsecond / 10 ** 6)
        target = monotonic() + slp_seconds
        handle = node.loop.call_later(slp_seconds, fire_time_event, target)

    @callback
    def fire_time_event(target) -> None:
        """Fire next time event."""
        now = dt.datetime.now(pytz.utc)
        node.events.publish_local(EVENT_TIME_CHANGED, {ATTR_NOW: now})

        # If we are more than a second late, a tick was missed
        ##late = monotonic() - target
        #if late > 1:
        #    node.events.async_fire(EVENT_TIMER_OUT_OF_SYNC, {ATTR_SECONDS: late})

        schedule_tick(now)

    @callback
    def stop_timer(_, __) -> None:
        """Stop the timer."""
        if handle is not None:
            _LOGGER.info("Timer is stopping.")
            handle.cancel()
    
    await node.events.async_listen(EVENT_NODE_STOP, stop_timer)

    _LOGGER.info("Timer is starting")
    schedule_tick(dt.datetime.now(pytz.utc))
