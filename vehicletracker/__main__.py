"""Start Vehicle Tracker."""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List

from vehicletracker.const import RESTART_EXIT_CODE, __version__
from vehicletracker.core import callback
from vehicletracker.helpers.config import load_config

class EventLoopPolicy(asyncio.DefaultEventLoopPolicy):  # type: ignore, pylint: disable=invalid-name
    """Event loop policy for Home Assistant."""

    def __init__(self, debug: bool) -> None:
        """Init the event loop policy."""
        super().__init__()
        self.debug = debug

    @property
    def loop_name(self) -> str:
        """Return name of the loop."""
        return self._loop_factory.__name__  # type: ignore

    def new_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop."""
        loop: asyncio.AbstractEventLoop = super().new_event_loop()
        loop.set_exception_handler(_async_loop_exception_handler)
        if self.debug:
            loop.set_debug(True)

        executor = ThreadPoolExecutor(thread_name_prefix="SyncWorker")
        loop.set_default_executor(executor)

        # Python 3.9+
        if hasattr(loop, "shutdown_default_executor"):
            return loop

        # Copied from Python 3.9 source
        def _do_shutdown(future: asyncio.Future) -> None:
            try:
                executor.shutdown(wait=True)
                loop.call_soon_threadsafe(future.set_result, None)
            except Exception as ex:  # pylint: disable=broad-except
                loop.call_soon_threadsafe(future.set_exception, ex)

        async def shutdown_default_executor() -> None:
            """Schedule the shutdown of the default executor."""
            future = loop.create_future()
            thread = threading.Thread(target=_do_shutdown, args=(future,))
            thread.start()
            try:
                await future
            finally:
                thread.join()

        setattr(loop, "shutdown_default_executor", shutdown_default_executor)

        return loop

@callback
def _async_loop_exception_handler(_: Any, context: Dict) -> None:
    """Handle all exception inside the core loop."""
    kwargs = {}
    exception = context.get("exception")
    if exception:
        kwargs["exc_info"] = (type(exception), exception, exception.__traceback__)

    logging.getLogger(__package__).error(
        "Error doing job: %s", context["message"], **kwargs  # type: ignore
    )

def get_arguments() -> argparse.Namespace:
    """Get parsed passed in arguments."""

    parser = argparse.ArgumentParser(
        description="Vehicle Tracker"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "-c",
        "--config",
        metavar="path_to_config_file",
        default="configuration.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging to file."
    )
    parser.add_argument(
        "--runner",
        action="store_true",
        help=f"On restart exit with code {RESTART_EXIT_CODE}",
    )
    if os.name == "posix":
        parser.add_argument(
            "--daemon", action="store_true", help="Run Home Assistant as daemon"
        )

    arguments = parser.parse_args()
    if os.name != "posix" or arguments.runner:
        setattr(arguments, "daemon", False)

    return arguments

def cmdline() -> List[str]:
    """Collect path and arguments to re-execute the current hass instance."""
    if os.path.basename(sys.argv[0]) == "__main__.py":
        modulepath = os.path.dirname(sys.argv[0])
        os.environ["PYTHONPATH"] = os.path.dirname(modulepath)
        return [sys.executable] + [arg for arg in sys.argv if arg != "--daemon"]

    return [arg for arg in sys.argv if arg != "--daemon"]

async def setup_and_run_node(config_file) -> int:
    """Set up node and run."""
    import logging
    from colorlog import ColoredFormatter

    # basicConfig must be called after importing colorlog in order to
    # ensure that the handlers it sets up wraps the correct streams.
    logging.basicConfig(level=logging.INFO)

    fmt = "%(asctime)s %(levelname)s (%(threadName)s) " "[%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    colorfmt = f"%(log_color)s{fmt}%(reset)s"
    logging.getLogger().handlers[0].setFormatter(
        ColoredFormatter(
            colorfmt,
            datefmt=datefmt,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        )
    )

    # If the above initialization failed for any reason, setup the default
    # formatting.  If the above succeeds, this wil result in a no-op.
    logging.basicConfig(format=fmt, datefmt=datefmt, level=logging.INFO)  

    # Suppress overly verbose logs from libraries that aren't helpful
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

    # pylint: disable=redefined-outer-name
    from vehicletracker import core
    
    config = load_config(config_file)
    node = core.VehicleTrackerNode(config)
    await core.async_setup_components(node, config)

    return await node.async_run()

def main() -> int:
    """Start Vehicle Tracker."""
    # Run a simple daemon runner process on Windows to handle restarts
    if os.name == "nt" and "--runner" not in sys.argv:
        nt_args = cmdline() + ["--runner"]
        while True:
            try:
                subprocess.check_call(nt_args)
                sys.exit(0)
            except KeyboardInterrupt:
                sys.exit(0)
            except subprocess.CalledProcessError as exc:
                if exc.returncode != RESTART_EXIT_CODE:
                    sys.exit(exc.returncode)

    args = get_arguments()
    config_file = os.path.join(os.getcwd(), args.config)

    asyncio.set_event_loop_policy(EventLoopPolicy(False))
    exit_code = asyncio.run(setup_and_run_node(config_file))
    #if exit_code == RESTART_EXIT_CODE and not args.runner:
    #    try_to_restart()

    return exit_code  # type: ignore

if __name__ == "__main__":
    sys.exit(main())
