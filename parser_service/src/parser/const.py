import re

TIMESTAMP_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6,7}Z)")
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

ERROR_KEYWORDS = ['ERROR', 'FAIL', 'EXCEPTION', 'TRACEBACK']
WARNING_KEYWORDS = ['WARNING', 'WARN']
DEBUG_KEYWORDS = ['DEBUG']
CRITICAL_KEYWORDS = ['CRITICAL', 'FATAL']