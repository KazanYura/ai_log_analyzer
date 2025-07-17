from abc import ABC, abstractmethod
from models.log_event import LogEvent
from typing import List

class BaseParser(ABC):
    @abstractmethod
    def parse(self, raw_log: str) -> List[LogEvent]:
        pass
