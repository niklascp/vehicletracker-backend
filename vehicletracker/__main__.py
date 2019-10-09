"""Start Vehicle Tracker."""

import sys
import os
import subprocess

from typing import List, Dict, Any, TYPE_CHECKING

import argparse
import asyncio

from vehicletracker.const import __version__, RESTART_EXIT_CODE
from vehicletracker.helpers.config import load_config

def set_loop() -> None:
    """Attempt to use uvloop."""    
    from asyncio.events import BaseDefaultEventLoopPolicy

    policy = None

    if sys.platform == "win32":
        if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
            # pylint: disable=no-member
            policy = asyncio.WindowsProactorEventLoopPolicy()
        else:

            class ProactorPolicy(BaseDefaultEventLoopPolicy):
                """Event loop policy to create proactor loops."""

                _loop_factory = asyncio.ProactorEventLoop

            policy = ProactorPolicy()
    else:
        try:
            import uvloop
        except ImportError:
            pass
        else:
            policy = uvloop.EventLoopPolicy()

    if policy is not None:
        asyncio.set_event_loop_policy(policy)

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
    set_loop()

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

    # Daemon functions
    #if args.pid_file:
    #    check_pid(args.pid_file)
    #if args.daemon:
    #    daemonize()
    #if args.pid_file:
    #    write_pid(args.pid_file)

    exit_code = asyncio.run(setup_and_run_node(config_file))
    #if exit_code == RESTART_EXIT_CODE and not args.runner:
    #    try_to_restart()

    return exit_code  # type: ignore

if __name__ == "__main__":
    sys.exit(main())
