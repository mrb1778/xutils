import inspect
import json
import typing

import numpy as np
import pandas as pd


def read(json_string):
    return json.loads(json_string)


def read_from(reader):
    return json.load(reader)


def read_file(path):
    with open(path) as file:
        return json.load(file)


def write(obj, pretty_print=False, indent_amount=2):
    indent = indent_amount if pretty_print else None
    obj = expand_json(obj)
    return json.dumps(obj, indent=indent)


def write_to(obj, writer=None, path=None, pretty_print=False, indent_amount=2):
    indent = indent_amount if pretty_print else None
    obj = expand_json(obj)
    if writer is None:
        with open(path, 'wt') as writer:
            json.dump(obj, writer, indent=indent)
            return path
    else:
        json.dump(obj, writer, indent=indent)
        return writer


class JsonExpander:
    def can_handle(self, obj):
        pass

    def expand(self, obj, default_expander):
        pass


json_handlers: typing.List[JsonExpander] = []


def json_handler(handler):
    json_handlers.append(handler())


@json_handler
class DictObjJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return hasattr(obj, "__dict__")

    def expand(self, obj, default_expander):
        return {
            key: default_expander(value)
            # key: value
            for key, value in inspect.getmembers(obj)
            if not key.startswith("__")
            and not inspect.isabstract(value)
            and not inspect.isbuiltin(value)
            and not inspect.isfunction(value)
            and not inspect.isgenerator(value)
            and not inspect.isgeneratorfunction(value)
            and not inspect.ismethod(value)
            and not inspect.ismethoddescriptor(value)
            and not inspect.isroutine(value)
        }


@json_handler
class ToJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return hasattr(obj, "to_json")

    def expand(self, obj, default_expander):
        return default_expander(obj.to_json())


@json_handler
class DictJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, dict)

    def expand(self, obj, default_expander):
        return {str(key): default_expander(value) for key, value in obj.items()}


@json_handler
class ListSetJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, (list, set, tuple))

    def expand(self, obj, default_expander):
        return list(map(default_expander, obj))


@json_handler
class NumpyJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, (np.ndarray, np.int64, np.float32, np.float64))

    def expand(self, obj, default_expander):
        return obj.tolist()


@json_handler
class PandasJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, (pd.DataFrame, pd.Series))

    def expand(self, obj, default_expander):
        return obj.to_dict()


def expand_json(obj):
    for handler in reversed(json_handlers):
        if handler.can_handle(obj):
            return handler.expand(obj, expand_json)

    return obj
