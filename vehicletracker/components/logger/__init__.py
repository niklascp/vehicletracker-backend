"""Support for setting the level of logging for components."""
import logging
import re

DOMAIN = "logger"

DATA_LOGGER = "logger"

LOGSEVERITY = {
    "CRITICAL": 50,
    "FATAL": 50,
    "ERROR": 40,
    "WARNING": 30,
    "WARN": 30,
    "INFO": 20,
    "DEBUG": 10,
    "NOTSET": 0,
}

LOGGER_DEFAULT = "default"
LOGGER_LOGS = "logs"

ATTR_LEVEL = "level"

class VehicleTrackerLogFilter(logging.Filter):
    """A log filter."""

    def __init__(self):
        """Initialize the filter."""
        super().__init__()

        self._default = None
        self._logs = None
        self._log_rx = None

    def update_default_level(self, default_level):
        """Update the default logger level."""
        self._default = default_level

    def update_log_filter(self, logs):
        """Rebuild the internal filter from new config."""
        #
        # A precompiled regex is used to avoid
        # the overhead of a list transversal
        #
        # Sort to make sure the longer
        # names are always matched first
        # so they take precedence of the shorter names
        # to allow for more granular settings.
        #
        names_by_len = sorted(list(logs), key=len, reverse=True)
        self._log_rx = re.compile("".join(["^(?:", "|".join(names_by_len), ")"]))
        self._logs = logs

    def filter(self, record):
        """Filter the log entries."""
        # Log with filtered severity
        if self._log_rx:
            match = self._log_rx.match(record.name)
            if match:
                return record.levelno >= self._logs[match.group(0)]

        # Log with default severity
        return record.levelno >= self._default


async def async_setup(hass, config):
    """Set up the logger component."""
    logfilter = {}
    vt_filter = VehicleTrackerLogFilter()

    def set_default_log_level(level):
        """Set the default log level for components."""
        logfilter[LOGGER_DEFAULT] = LOGSEVERITY[level.upper()]
        vt_filter.update_default_level(LOGSEVERITY[level.upper()])

    def set_log_levels(logpoints):
        """Set the specified log levels."""
        logs = {}

        # Preserve existing logs
        if LOGGER_LOGS in logfilter:
            logs.update(logfilter[LOGGER_LOGS])

        # Add new logpoints mapped to correct severity
        for key, value in logpoints.items():
            logs[key] = LOGSEVERITY[value.upper()]

        logfilter[LOGGER_LOGS] = logs

        vt_filter.update_log_filter(logs)

    # Set default log severity
    if LOGGER_DEFAULT in config.get(DOMAIN):
        set_default_log_level(config.get(DOMAIN)[LOGGER_DEFAULT])
    else:
        set_default_log_level("DEBUG")

    logger = logging.getLogger("")
    logger.setLevel(logging.NOTSET)

    # Set log filter for all log handler
    for handler in logging.root.handlers:
        handler.setLevel(logging.NOTSET)
        handler.addFilter(vt_filter)

    if LOGGER_LOGS in config.get(DOMAIN):
        set_log_levels(config.get(DOMAIN)[LOGGER_LOGS])

    return True