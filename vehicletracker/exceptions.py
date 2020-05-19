class ApplicationError(Exception):
   """Base class for other exceptions"""
   pass

class ModelNotFound(ApplicationError):
   """Raised when no model meeting the specification could be found."""
   def __init__(self, message):
      self.message = message
