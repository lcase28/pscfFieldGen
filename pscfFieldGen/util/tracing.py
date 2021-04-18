"""
Module contains classes and methods for tracing program operation.
"""
from enum import IntEnum

class TraceLevel(IntEnum):
    DEBUG = 150
    DETAIL = 100
    ALL = 100
    RESULT = 80
    EVENT = 50
    READ = 25
    ECHO = 20
    NONE = 0

_indent_strings = { TraceLevel.NONE   : "{}",
                    TraceLevel.ECHO   : "ECHO   : {}",
                    TraceLevel.READ   : "READ   : {}",
                    TraceLevel.EVENT  : "EVENT  : {}",
                    TraceLevel.RESULT : "RESULT : {}",
                    TraceLevel.ALL    : "DETAIL : {}",
                    TraceLevel.DETAIL : "DETAIL : {}",
                    TraceLevel.DEBUG  : "DEBUG  : {}" }

class TraceManager:
    """
    Class to manage trace output, depending on set level.
    """
    
    def __init__(self, level=TraceLevel.NONE):
        self._filter = level
    
    @property
    def filterLevel(self):  
        return self._filter
    
    @filterLevel.setter
    def filterLevel(self, val):
        self._filter = val
    
    def passesFilter(self, filter_level):
        """
        Return True if a message at filter_level would be printed.
        """
        return filter_level <= self._filter
    
    def trace(self, message, level, *args):
        if level <= self._filter:
            if len(args) > 0:
                message = message.format(*args)
            formstr = _indent_strings.get(level,"\t\tOTHER: {}")
            print(formstr.format(message))

TRACER = TraceManager(TraceLevel.NONE)

def debug(caller_name, msg, *args):
    if TRACER.passesFilter(TraceLevel.DEBUG):
        msg = caller_name + ": " + msg
        TRACER.trace(msg, TraceLevel.DEBUG, *args)
