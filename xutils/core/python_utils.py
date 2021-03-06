import os
import sys
import inspect
from typing import Iterable


class DisablePrintStatements:
    """with DisablePrintStatements(): ..."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_params(fun) -> dict:
    sig = inspect.signature(fun)
    return {p.name: p.default
            for p in sig.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default != inspect.Parameter.empty}


def remove_keys(items: dict, keys: Iterable[str]):
    for key in keys:
        if key in items:
            del items[key]


def getattr_ignore_case(obj, attr: str):
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)
