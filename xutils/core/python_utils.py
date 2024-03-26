import importlib
import itertools
from argparse import ArgumentParser
from typing import List
import argparse
import collections
import gc
import os
import sys
import inspect
from typing import Iterable, Dict, Any, Callable, get_type_hints, Optional, Union
from tabulate import tabulate


class DisablePrintStatements:
    """with DisablePrintStatements(): ..."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_functions(obj=None, module=None, private=False, as_dict=False):
    if obj is None and module is None:
        raise ValueError("Need obj or module")
    if obj is None and module is not None and isinstance(module, str):
        obj = importlib.import_module(module)
    elif obj is None and module is not None:
        obj = module

    functions = inspect.getmembers(obj, inspect.isfunction)
    if not private:
        functions = [(key, value)
                     for (key, value) in functions
                     if not key.startswith("_")]
    if as_dict:
        return {key: value
                for (key, value) in functions}
    else:
        return functions


def params_has_kwargs(fun: Callable) -> bool:
    for p in params_sig(fun):
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def param_names(fun: Callable, required=True, optional=True) -> list:
    return [
        p.name
        for p in params_sig(fun)
        if (required and p.default == inspect.Parameter.empty) or
           (optional and p.kind != inspect.Parameter.empty)
    ]


def param_defaults(fun: Callable, required=True, optional=True) -> dict:
    return {
        p.name: p.default if p.default != inspect.Parameter.empty else None
        for p in params_sig(fun)
        if (required and p.default == inspect.Parameter.empty) or
           (optional and p.kind != inspect.Parameter.empty)
    }


def params_sig(fun: Callable):
    return inspect.signature(fun).parameters.values()


def params(fun: Callable, required=True, optional=True) -> Dict[str, Dict[str, Any]]:
    return {
        p.name: {
            "name": p.name,
            "default": p.default if p.default != inspect.Parameter.empty else None,
            "required": p.default == inspect.Parameter.empty,
            "type": p.annotation if p.annotation != inspect.Parameter.empty else None,
            "positional": p.kind == inspect.Parameter.POSITIONAL_ONLY,
            "args": p.kind == inspect.Parameter.VAR_POSITIONAL,
            "kwargs": p.kind == inspect.Parameter.VAR_KEYWORD
        }
        for p in params_sig(fun)
        if (required and p.default == inspect.Parameter.empty) or
           (optional and p.kind != inspect.Parameter.empty)
    }


def return_type(fun: Callable) -> Optional[Union[type, str]]:
    if not callable(fun):
        return None

    hints = get_type_hints(fun)
    return hints.get("return")


def get_package(package: str):
    return importlib.import_module(package)


def execute(fun: str,
            package: Optional[str] = None,
            obj: Optional[Any] = None,
            args: Optional[List] = None,
            kwargs: Optional[Dict] = None) -> Any:
    if obj is not None:
        if not hasattr(obj, fun):
            raise ValueError(
                f"Object: '{obj}':{type(obj)} does not have function: '{fun}' options: {get_functions(obj)}")
        fun_exec = getattr(obj, fun)
    elif package is not None:
        module = get_package(package)
        if not hasattr(module, fun):
            raise ValueError(
                f"Module: '{module}' does not have function: '{fun}' options: {get_functions(module=module)}")
        fun_exec = getattr(module, fun)
    else:
        if fun not in globals():
            raise ValueError(f"{fun} is not a built in function")
        fun_exec = globals()["__builtins__"][fun]

    if args is None:
        if kwargs is None:
            return fun_exec()
        else:
            return fun_exec(**kwargs)
    elif kwargs is None:
        return fun_exec(*args)
    else:
        return fun_exec(*args, **kwargs)


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
    if sort_on is not None:
        dicts = sorted(dicts, key=lambda entry: entry[sort_on], reverse=True)
    header = dicts[0].keys()
    rows = [x.values() for x in dicts]
    output = tabulate(rows, header)
    print(output)
    return output


def bordered(*text):
    return tabulate([text], tablefmt='fancy_grid')


def print_bordered(*text):
    print(bordered(*text))


def print_dict(d: dict, key_header="Key", value_header="Value"):
    output = tabulate(d.items(), [key_header, value_header], tablefmt='fancy_grid')
    print(output)
    return output


def print_list_dict(data: list, keys=None):
    if keys is None:
        keys = data[0].keys()

    output = tabulate([list(map(x.get, keys)) for x in data], keys)
    print(bordered(output))
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


class ParseKwargs:
    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = {}
        for value in values:
            key, value = value.split('=')
            kwargs[key] = value
        setattr(namespace, 'kwargs', kwargs)


def add_kwargs_arg(parser: ArgumentParser, arg_name: str = "args") -> None:
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
    return zip(*[iter(iterable)] * n)


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
    import six
    return isinstance(arg, collections.Iterable) and not isinstance(arg, six.string_types)


class DictAccess(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def permute_transpose(data: Dict[Any, Iterable[Any]]) -> List[Dict]:
    keys, values = zip(*data.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def curry_kwargs(fun: Callable, **kwargs):
    return lambda **extra_kwargs: fun(
        **{**kwargs, **extra_kwargs}
    )


def curry_args_to_kwargs(fun: Callable, **kwargs):
    def args_kwargs_fun(*args, **extra_kwargs):
        all_params = list(params(fun, required=True, optional=False).keys())
        args_kwargs = {all_params[i]: p for i, p in enumerate(args)}
        return fun(**{**kwargs, **extra_kwargs, **args_kwargs})

    return args_kwargs_fun


def walk_dict(dic, visitor: Callable, path=()):
    to_process = [(dic, path)]
    while to_process:
        dict_node, path_node = to_process.pop(0)
        for key, value in dict_node.items():
            visitor(dict_node, key, value)
            if isinstance(value, dict):
                to_process.append((value, path_node + (key,)))
