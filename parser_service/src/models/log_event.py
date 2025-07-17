from dataclasses import dataclass
from enum import Enum
from typing import Optional

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEvent:
    timestamp: Optional[str]
    level: Optional[LogLevel]
    message: str
    source: str = "circleci"