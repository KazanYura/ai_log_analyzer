from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEvent(BaseModel):
    timestamp: Optional[str]
    level: Optional[LogLevel]
    message: str
    source: Optional[str] = "circleci"


class AdviceRequest(BaseModel):
    events: List[LogEvent]


class AdviceResponse(BaseModel):
    advice: str
