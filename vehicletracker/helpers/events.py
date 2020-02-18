"""Helpers for listening to events."""
from datetime import datetime, timedelta
import functools as ft
from typing import Any, Callable, Dict, Iterable, Optional, Union, cast

from vehicletracker.core import CALLBACK_TYPE, VehicleTrackerNode, callback
from vehicletracker.helpers import datetime as dt_util
from vehicletracker.helpers.async_ import run_callback_threadsafe

from vehicletracker.const import ATTR_NOW, EVENT_TIME_CHANGED

def threaded_listener_factory(async_factory: Callable[..., Any]) -> CALLBACK_TYPE:
    """Convert an async event helper to a threaded one."""

    @ft.wraps(async_factory)
    def factory(*args: Any, **kwargs: Any) -> CALLBACK_TYPE:
        """Call async event helper safely."""
        node = args[0]

        if not isinstance(node, VehicleTrackerNode):
            raise TypeError("First parameter needs to be a Vehicle Tracker Node instance")

        async_remove = run_callback_threadsafe(
            node.loop, ft.partial(async_factory, *args, **kwargs)
        ).result()

        def remove() -> None:
            """Threadsafe removal."""
            run_callback_threadsafe(node.loop, async_remove).result()

        return remove

    return factory

@callback
async def async_track_utc_time_change(
    node: VehicleTrackerNode,
    action: Callable[..., None],
    hour: Optional[Any] = None,
    minute: Optional[Any] = None,
    second: Optional[Any] = None,
    local: bool = False,
) -> CALLBACK_TYPE:
    """Add a listener that will fire if time matches a pattern."""
    # We do not have to wrap the function with time pattern matching logic
    # if no pattern given
    matching_seconds = dt_util.parse_time_expression(second, 0, 59)
    matching_minutes = dt_util.parse_time_expression(minute, 0, 59)
    matching_hours = dt_util.parse_time_expression(hour, 0, 23)

    next_time = None

    def calculate_next(now: datetime) -> None:
        """Calculate and set the next time the trigger should fire."""
        nonlocal next_time

        localized_now = dt_util.as_local(now) if local else now
        next_time = dt_util.find_next_time_expression_time(
            localized_now, matching_seconds, matching_minutes, matching_hours
        )

    # Make sure rolling back the clock doesn't prevent the timer from
    # triggering.
    last_now: Optional[datetime] = None

    @callback
    def pattern_time_change_listener(event_data) -> None:
        """Listen for matching time_changed events."""
        nonlocal next_time, last_now

        now = event_data[ATTR_NOW]

        if last_now is None or now < last_now:
            # Time rolled back or next time not yet calculated
            calculate_next(now)

        last_now = now

        if next_time <= now:
            node.async_run_job(action, dt_util.as_local(now) if local else now)
            calculate_next(now + timedelta(seconds=1))

    # We can't use async_track_point_in_utc_time here because it would
    # break in the case that the system time abruptly jumps backwards.
    # Our custom last_now logic takes care of resolving that scenario.
    return await node.events.async_listen(EVENT_TIME_CHANGED, pattern_time_change_listener)

track_utc_time_change = threaded_listener_factory(async_track_utc_time_change)
