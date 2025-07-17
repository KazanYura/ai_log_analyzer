from functools import wraps
from fastapi import HTTPException
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def ai_service_exception_handler(func: Callable) -> Callable:
    """
    Decorator that handles common exceptions for AI service operations and converts them to FastAPI HTTPExceptions.
    
    Handles:
    - ValueError -> 400 Bad Request
    - ConnectionError -> 503 Service Unavailable
    - TimeoutError -> 504 Gateway Timeout
    - General exceptions -> 500 Internal Server Error
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid input: {str(e)}"
            )
        except ConnectionError:
            raise HTTPException(
                status_code=503, 
                detail="AI service unavailable"
            )
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to {func.__name__.replace('_', ' ')} {e}"
            )
    return wrapper


def health_check_exception_handler(func: Callable) -> Callable:
    """
    Decorator specifically for health check endpoints.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in health_check: {e}")
            raise HTTPException(status_code=500, detail="Health check failed")
    return wrapper
