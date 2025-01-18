
import signal
from functools import wraps
from contextlib import contextmanager
import time

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Experiment timed out")
    
    # Register signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable alarm
        signal.alarm(0)