"""Support to serve the Vehicle Tracker API as WSGI application."""
from ipaddress import ip_network
import logging
import os
import json
from typing import Optional

import asyncio
import async_timeout
from aiohttp import web
from aiohttp.web_exceptions import HTTPMovedPermanently

DOMAIN = "http"

_LOGGER = logging.getLogger(__name__)

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 5000

STREAM_PING_EVENT = { 'eventType': 'ping' }
STREAM_PING_INTERVAL = 50

async def async_setup(node, config):
    """Set up the HTTP API."""
    conf = config.get(DOMAIN) or {}

    server = VehicleTrackerHTTP(
        node,
        server_host=conf.get('server_host', DEFAULT_SERVER_HOST),
        server_port=conf.get('server_port', DEFAULT_SERVER_PORT),
        cors_origins=conf.get('cors_origins', []),
    )

    async def stop_server(event):
        """Stop the server."""
        await server.stop()

    async def start_server(event):
        """Start the server."""
        #hass.bus.async_listen_once(EVENT_VEHICLETRACKER_STOP, stop_server)
        await server.start()

    #node.event.async_listen_once(EVENT_VEHICLETRACKER_START, start_server)

    node.http = server

    async def service(request):
        service = request.match_info['service']
        service_data = {}
        for k, v in request.rel_url.query.items():
            service_data[k] = v
        result = await node.services.async_call(
            service, service_data
        )
        return web.json_response(result)

    server.app.router.add_route('get', '/api/services/{service}', service)

    async def event_stream(request):
        buffer = asyncio.Queue() 

        async def forward_events(event):
            await buffer.put(event)

        response = web.StreamResponse()
        response.content_type = "text/event-stream"
        await response.prepare(request)

        unsub_stream = await node.events.async_listen(
            request.query.get('event_type', '*'),
            forward_events)

        try:
            _LOGGER.debug("STREAM %s ATTACHED", id(buffer))

            # Fire off one message so browsers fire open event right away
            await forward_events(STREAM_PING_EVENT)

            while True:
                try:
                    with async_timeout.timeout(STREAM_PING_INTERVAL):
                        payload = await buffer.get()

                    msg = f"data: {payload}\n\n"
                    _LOGGER.debug("STREAM %s WRITING %s", id(buffer), msg.strip())
                    await response.write(msg.encode("UTF-8"))
                except asyncio.TimeoutError:
                    await forward_events(STREAM_PING_EVENT)

        except asyncio.CancelledError:
            _LOGGER.debug("STREAM %s ABORT", id(buffer))

        finally:
            _LOGGER.debug("STREAM %s RESPONSE CLOSED", id(buffer))
            unsub_stream()

        return response

    server.app.router.add_route('get', '/api/event-stream', event_stream)

    await server.start()

    return True

class VehicleTrackerHTTP:
    """HTTP server for Vehicle Tracker."""

    def __init__(
        self,
        node,
        server_host,
        server_port,
        cors_origins,
    ):
        """Initialize the HTTP Home Assistant server."""
        app = self.app = web.Application(middlewares=[])

        #setup_cors(app, cors_origins)

        self.node = node
        self.server_host = server_host
        self.server_port = server_port
        self._handler = None
        self.runner = None
        self.site = None

    async def start(self):
        """Start the aiohttp server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(
            self.runner, self.server_host, self.server_port)
        try:
            await self.site.start()
        except OSError as error:
            _LOGGER.error(
                "Failed to create HTTP server at port %d: %s", self.server_port, error
            )

    async def stop(self):
        """Stop the aiohttp server."""
        await self.site.stop()
        await self.runner.cleanup()
