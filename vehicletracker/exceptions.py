class ApplicationError(Exception):
   """Base class for other exceptions"""
   pass

class ModelNotFound(ApplicationError):
   """Base class for other exceptions"""
   def __init__(self, message):
      self.message = message
