import time
import random
import logging
from functools import wraps
from src.utils.config import config

logger = logging.getLogger(__name__)

def retry_with_backoff(exceptions=(Exception,), max_attempts=None, backoff_factor=None, max_backoff=None):
    """
    Decorator that retries a function with exponential backoff.
    Uses configurations from config if values are not provided.
    """
    if max_attempts is None:
        max_attempts = config.retry_attempts
    if backoff_factor is None:
        backoff_factor = config.retry_backoff_factor
    if max_backoff is None:
        max_backoff = config.retry_max_backoff

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts. Error: {e}",
                            exc_info=True
                        )
                        raise e
                    
                    # Calculate wait time: factor * (2 ** (attempt - 1)) + jitter
                    wait_time = backoff_factor * (2 ** (attempts - 1))
                    wait_time = min(wait_time, max_backoff)
                    jitter = random.uniform(0.1, 0.5)
                    total_wait = wait_time + jitter
                    
                    logger.warning(
                        f"Attempt {attempts} for {func.__name__} failed: {e}. "
                        f"Retrying in {total_wait:.2f} seconds..."
                    )
                    time.sleep(total_wait)
            return func(*args, **kwargs)
        return wrapper
    return decorator
