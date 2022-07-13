import argparse
import collections
import gc
import os
import sys
import inspect
from types import FunctionType
from typing import Iterable

import six


class DisablePrintStatements:
    """with DisablePrintStatements(): ..."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def params_has_kwargs(fun: FunctionType) -> bool:
    sig = inspect.signature(fun)
    for p in sig.parameters.values():
        return p.kind == inspect.Parameter.VAR_KEYWORD


def param_names(fun: FunctionType, required=True, optional=True) -> list:
    sig = inspect.signature(fun)
    return [
        p.name
        for p in sig.parameters.values()
        if (required and p.default == inspect.Parameter.empty)
        or (optional and p.kind != inspect.Parameter.empty)
    ]


def param_defaults(fun: FunctionType, required=True, optional=True) -> dict:
    sig = inspect.signature(fun)
    return {
        p.name: p.default if p.default != inspect.Parameter.empty else None
        for p in sig.parameters.values()
        if (required and p.default == inspect.Parameter.empty)
        or (optional and p.kind != inspect.Parameter.empty)
    }


def params(fun: FunctionType, required=True, optional=True) -> dict:
    sig = inspect.signature(fun)
    return {
        p.name: {
            "name": p.name,
            "default": p.default if p.default != inspect.Parameter.empty else None,
            "required": p.default == inspect.Parameter.empty,
            "type": p.annotation if p.annotation != inspect.Parameter.empty else None
        }
        for p in sig.parameters.values()
        if (required and p.default == inspect.Parameter.empty)
        or (optional and p.kind != inspect.Parameter.empty)
    }


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


def print_dict(d: dict, key_header="Key", value_header="Value", ignore_none=False):
    import tabulate

    output = tabulate.tabulate(d.items(), [key_header, value_header], tablefmt='fancy_grid')
    print(output)
    return output


def print_list_dict(data: list, keys=None):
    import tabulate

    if keys is None:
        keys = data[0].keys()

    output = tabulate.tabulate([list(map(x.get, keys)) for x in data], keys)
    print(output)
    return output


def remove_na_from_dict(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = remove_na_from_dict(v)
            if len(nested.keys()) > 0:
                clean[k] = nested
        elif v is not None:
            clean[k] = v
    return clean


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


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def add_kwargs_arg(parser, arg_name="args"):
    parser.add_argument(arg_name, nargs='*', action=ParseKwargs)


def parse_unknown_args(parser):
    args, unknown_args = parser.parse_known_args()
    args_dict = vars(args)
    unknown_args_dict = {}
    for key, value in grouped(unknown_args, 2):
        if key.startswith("--"):
            key = key[2:]
        elif key.startswith("-"):
            key = key[1:]

        unknown_args_dict[key] = value

    return {**args_dict, **unknown_args_dict}


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)


def safe_eval(code: str, params: dict = None):
    safe_list = ["__name__",
                 "__doc__",
                 "__package__",
                 "__loader__",
                 "__spec__",
                 "__build_class__",
                 "__import__",
                 "abs",
                 "all",
                 "any",
                 "ascii",
                 "bin",
                 "breakpoint",
                 "callable",
                 "chr",
                 "compile",
                 "delattr",
                 "dir",
                 "divmod",
                 # "eval",
                 # "exec",
                 "format",
                 "getattr",
                 "globals",
                 "hasattr",
                 "hash",
                 "hex",
                 "id",
                 "input",
                 "isinstance",
                 "issubclass",
                 "iter",
                 "len",
                 "locals",
                 "max",
                 "min",
                 "next",
                 "oct",
                 "ord",
                 "pow",
                 "print",
                 "repr",
                 "round",
                 "setattr",
                 "sorted",
                 "sum",
                 "vars",
                 "None",
                 "Ellipsis",
                 "NotImplemented",
                 "False",
                 "True",
                 "bool",
                 "memoryview",
                 "bytearray",
                 "bytes",
                 "classmethod",
                 "complex",
                 "dict",
                 "enumerate",
                 "filter",
                 "float",
                 "frozenset",
                 "property",
                 "int",
                 "list",
                 "map",
                 "object",
                 "range",
                 "reversed",
                 "set",
                 "slice",
                 "staticmethod",
                 "str",
                 "super",
                 "tuple",
                 "type",
                 "zip",
                 "__debug__",
                 "BaseException",
                 "Exception",
                 "TypeError",
                 "StopAsyncIteration",
                 "StopIteration",
                 "GeneratorExit",
                 "SystemExit",
                 "KeyboardInterrupt",
                 "ImportError",
                 "ModuleNotFoundError",
                 "OSError",
                 "EnvironmentError",
                 "IOError",
                 "EOFError",
                 "RuntimeError",
                 "RecursionError",
                 "NotImplementedError",
                 "NameError",
                 "UnboundLocalError",
                 "AttributeError",
                 "SyntaxError",
                 "IndentationError",
                 "TabError",
                 "LookupError",
                 "IndexError",
                 "KeyError",
                 "ValueError",
                 "UnicodeError",
                 "UnicodeEncodeError",
                 "UnicodeDecodeError",
                 "UnicodeTranslateError",
                 "AssertionError",
                 "ArithmeticError",
                 "FloatingPointError",
                 "OverflowError",
                 "ZeroDivisionError",
                 "SystemError",
                 "ReferenceError",
                 "MemoryError",
                 "BufferError",
                 "Warning",
                 "UserWarning",
                 "DeprecationWarning",
                 "PendingDeprecationWarning",
                 "SyntaxWarning",
                 "RuntimeWarning",
                 "FutureWarning",
                 "ImportWarning",
                 "UnicodeWarning",
                 "BytesWarning",
                 "ResourceWarning",
                 "ConnectionError",
                 "BlockingIOError",
                 "BrokenPipeError",
                 "ChildProcessError",
                 "ConnectionAbortedError",
                 "ConnectionRefusedError",
                 "ConnectionResetError",
                 "FileExistsError",
                 "FileNotFoundError",
                 "IsADirectoryError",
                 "NotADirectoryError",
                 "InterruptedError",
                 "PermissionError",
                 "ProcessLookupError",
                 "TimeoutError",
                 # "open",
                 # "quit",
                 # "exit",
                 "copyright",
                 "credits",
                 "license",
                 "help"]
    safe_locals = dict([(k, locals().get(k, None)) for k in safe_list])

    local_params = {**safe_locals, **params} if params is not None else safe_locals
    return eval(code, {"__builtins__": None}, local_params)


def free_memory():
    gc.collect()


def iterable(arg):
    return isinstance(arg, collections.Iterable) and not isinstance(arg, six.string_types)

