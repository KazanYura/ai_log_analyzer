from .circleci_parser import CircleCIParser
from .base import BaseParser

def get_parser(source: str) -> BaseParser:
    if source == "circleci":
        return CircleCIParser()
    raise ValueError(f"Unsupported log source: {source}")
