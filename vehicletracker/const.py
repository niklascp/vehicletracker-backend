MAJOR_VERSION = 0
MINOR_VERSION = 1
PATCH_VERSION = "0-beta"
__short_version__ = "{}.{}".format(MAJOR_VERSION, MINOR_VERSION)
__version__ = "{}.{}".format(__short_version__, PATCH_VERSION)

# The exit code to send to request a restart
RESTART_EXIT_CODE = 100