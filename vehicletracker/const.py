MAJOR_VERSION = 0
MINOR_VERSION = 34
PATCH_VERSION = ""
__short_version__ = "{}.{}".format(MAJOR_VERSION, MINOR_VERSION)
__version__ = "{}{}".format(__short_version__, PATCH_VERSION)

# The exit code to send to request a restart
RESTART_EXIT_CODE = 100

MATCH_ALL = '*'

ATTR_EVENT_TYPE = 'eventType'
ATTR_NODE_NAME = 'node'
ATTR_NOW = 'now'

EVENT_NODE_START = 'node_start'
EVENT_NODE_STOP = 'node_stop'
EVENT_REPLY = 'reply'
EVENT_TIME_CHANGED = 'time_changed'

# How long to wait till things that run on startup have to finish.
TIMEOUT_EVENT_START = 15
# How long to wait till things that run on shutdown have to finish.
TIMEOUT_EVENT_STOP = 15
