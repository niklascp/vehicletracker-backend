"""Signal handling related helpers."""
import logging
import signal
import sys
from types import FrameType

from vehicletracker.core import VehicleTrackerNode
from vehicletracker.const import RESTART_EXIT_CODE

_LOGGER = logging.getLogger(__name__)

def async_register_signal_handling(node: VehicleTrackerNode) -> None:
    """Register system signal handler for core."""
    if sys.platform != "win32":

        def async_signal_handle(exit_code: int) -> None:
            """Wrap signal handling.
            * queue call to shutdown task
            * re-instate default handler
            """
            node.loop.remove_signal_handler(signal.SIGTERM)
            node.loop.remove_signal_handler(signal.SIGINT)
            node.async_create_task(node.async_stop(exit_code))

        try:
            node.loop.add_signal_handler(signal.SIGTERM, async_signal_handle, 0)
        except ValueError:
            _LOGGER.warning("Could not bind to SIGTERM")

        try:
            node.loop.add_signal_handler(signal.SIGINT, async_signal_handle, 0)
        except ValueError:
            _LOGGER.warning("Could not bind to SIGINT")

        try:
            node.loop.add_signal_handler(
                signal.SIGHUP, async_signal_handle, RESTART_EXIT_CODE
            )
        except ValueError:
            _LOGGER.warning("Could not bind to SIGHUP")

    else:
        old_sigterm = None
        old_sigint = None

        def async_signal_handle(exit_code: int, frame: FrameType) -> None:
            """Wrap signal handling.
            * queue call to shutdown task
            * re-instate default handler
            """
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.signal(signal.SIGINT, old_sigint)
            node.async_create_task(node.async_stop(exit_code))

        try:
            old_sigterm = signal.signal(signal.SIGTERM, async_signal_handle)
        except ValueError:            
            _LOGGER.warning("Could not bind to SIGTERM")

        try:
            old_sigint = signal.signal(signal.SIGINT, async_signal_handle)
        except ValueError:            
            _LOGGER.warning("Could not bind to SIGINT")
