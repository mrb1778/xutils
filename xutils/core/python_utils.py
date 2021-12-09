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


def merge_lists(base, overrides, attr="name"):
    # Merge two lists of classes by comparing them for equality using 'attr'.
    # This function prefers anything in 'overrides'. In other words, if a class
    # is present in overrides and matches (according to the equality criterion) a class in
    # base, it will be used instead of the one in base.
    merged = list(overrides)
    existing = set([getattr(o, attr) for o in overrides])
    merged.extend([d for d in base if getattr(d, attr) not in existing])
    return merged


def getattr_ignore_case(obj, attr: str):
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


def print_dicts(dicts, sort_on=None):
    import tabulate

    if sort_on is not None:
        dicts = sorted(dicts, key=lambda entry: entry[sort_on], reverse=True)
    header = dicts[0].keys()
    rows = [x.values() for x in dicts]
    output = tabulate.tabulate(rows, header)
    print(output)
    return output


def print_dict(d: dict, key_header="Key", value_header="Value"):
    import tabulate

    output = tabulate.tabulate(d.items(), [key_header, value_header])
    print(output)
    return output


def get_python_version():
    import platform

    versions = {"2": "2.7.15", "3": "3.7.3"}
    return versions[platform.python_version_tuple()[0]]


def delete_from_globals(*args):
    for _delete in args:
        try:
            del globals()[_delete]
        except KeyError:
            pass
    try:
        del globals()["_delete"]
    except KeyError:
        pass


def get_module(name):
    return sys.modules[name]


def find_in_docstring(module, look_for):
    with_docstring = []
    for element_name in dir(module):
        element = getattr(module, element_name)
        if hasattr(element, '__call__'):
            if not inspect.isbuiltin(element):
                doc = str(element.__doc__)
                if doc is not None and look_for in doc:
                    with_docstring.append(element)
    return with_docstring


def get_stack(only=None, print_=True):
    stack = []
    stack_frame = inspect.currentframe()
    while stack_frame:
        if only is None or any(only in s for s in [stack_frame.f_code.co_name, stack_frame.f_code.co_filename]):
            call = {
                "file": stack_frame.f_code.co_filename,
                "fun": stack_frame.f_code.co_name,
                "line": stack_frame.f_lineno,
            }
            stack.append(call)
            if print_:
                print(call["fun"], ' ', call["file"], ":", call["line"], sep='')
        if stack_frame.f_code.co_name == '<module>':
            if stack_frame.f_code.co_filename != '<stdin>':
                caller_module = inspect.getmodule(stack_frame)
            else:
                caller_module = sys.modules['__main__']
            if not caller_module is None and print_:
                print("******BEGIN******")
            break
        stack_frame = stack_frame.f_back
        return stack