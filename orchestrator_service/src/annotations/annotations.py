from functools import wraps
from fastapi import HTTPException
import httpx
import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


def exception_handler(
    timeout_status: int = 504,
    timeout_message: str = "Service timeout",
    http_error_status: int = 503,
    http_error_message: str = "Service unavailable",
    general_error_status: int = 500,
    general_error_message: str = "Service error",
    service_name: str = "Service"
):
    """
    Decorator that handles common HTTP client exceptions and converts them to FastAPI HTTPExceptions.
    
    Args:
        timeout_status: HTTP status code for timeout errors
        timeout_message: Error message for timeout errors
        http_error_status: HTTP status code for HTTP errors
        http_error_message: Error message for HTTP errors
        general_error_status: HTTP status code for general errors
        general_error_message: Error message for general errors
        service_name: Name of the service for logging purposes
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except httpx.TimeoutException:
                logger.error(f"{service_name} timeout")
                raise HTTPException(status_code=timeout_status, detail=f"{service_name} {timeout_message}")
            except httpx.HTTPError as e:
                logger.error(f"{service_name} error: {e}")
                raise HTTPException(status_code=http_error_status, detail=f"{service_name} {http_error_message}")
            except Exception as e:
                logger.error(f"Unexpected error calling {service_name}: {e}")
                raise HTTPException(status_code=general_error_status, detail=f"Failed to {general_error_message}")
        return wrapper
    return decorator


def parser_service_exception_handler(func: Callable) -> Callable:
    """Specific exception handler for parser service calls."""
    return exception_handler(
        timeout_message="timeout",
        http_error_message="unavailable",
        general_error_message="parse logs",
        service_name="Parser service"
    )(func)


def ai_analyzer_service_exception_handler(func: Callable) -> Callable:
    """Specific exception handler for AI analyzer service calls."""
    return exception_handler(
        timeout_message="timeout",
        http_error_message="unavailable", 
        general_error_message="analyze events",
        service_name="AI analyzer service"
    )(func)
