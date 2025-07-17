from typing import List, Optional
import urllib.parse
from .base import BaseParser
from models.log_event import LogEvent, LogLevel
from .const import (
    ANSI_ESCAPE,
    CRITICAL_KEYWORDS, 
    DEBUG_KEYWORDS, 
    ERROR_KEYWORDS, 
    TIMESTAMP_PATTERN,
    WARNING_KEYWORDS
)

class CircleCIParser(BaseParser):
    def parse(self, raw_log: str) -> List[LogEvent]:
        log_lines = raw_log.strip().split('\n')
        events = []
        
        for line_num, line in enumerate(log_lines):
            if not line.strip():
                continue
                
            timestamp = self._extract_timestamp(line)
            
            level = self._determine_log_level(line)
            
            message = self._extract_message(line)

            event = LogEvent(
                timestamp=timestamp,
                level=level,
                message=message,
                source="circleci"
            )
            events.append(event)
        
        return events
    
    def _extract_timestamp(self, line: str) -> Optional[str]:
        match = TIMESTAMP_PATTERN.search(line)
        return match.group(1) if match else None
    
    def _determine_log_level(self, line: str) -> LogLevel:
        line_upper = line.upper()
        
        if any(keyword in line_upper for keyword in ERROR_KEYWORDS):
            return LogLevel.ERROR
        elif any(keyword in line_upper for keyword in WARNING_KEYWORDS):
            return LogLevel.WARNING
        elif any(keyword in line_upper for keyword in DEBUG_KEYWORDS):
            return LogLevel.DEBUG
        elif any(keyword in line_upper for keyword in CRITICAL_KEYWORDS):
            return LogLevel.CRITICAL
        else:
            return LogLevel.INFO
    
    def _extract_message(self, line: str) -> str:
        timestamp = self._extract_timestamp(line)
        if timestamp:
            line = line.replace(timestamp, '')
        line = line.strip()
        line = self._strip_ansi_codes(line)
        return urllib.parse.unquote(line)

    
    @staticmethod
    def _strip_ansi_codes(line: str) -> str:
        return ANSI_ESCAPE.sub('', line)