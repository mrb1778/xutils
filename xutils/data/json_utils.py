import functools
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

    def expand(self, obj):
        pass


json_handlers: typing.List[JsonExpander] = []


# @functools.wraps
def json_handler(handler):
    handler_instance = handler()
    json_handlers.append(handler_instance)
    return handler_instance


@json_handler
class DictObjJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return hasattr(obj, "__dict__")

    def expand(self, obj):
        return {
            key: expand_json(value)
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
class ToStringHiddenExpander(JsonExpander):
    def can_handle(self, obj):
        return hasattr(obj, "__toString")

    def expand(self, obj):
        return expand_json(obj.__toString())


@json_handler
class ToJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return hasattr(obj, "to_json")

    def expand(self, obj):
        return expand_json(obj.to_json())


@json_handler
class DictJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, dict)

    def expand(self, obj):
        return {str(key):
                expand_json(value)
                for key, value in obj.items()}


@json_handler
class ListSetJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, (list, set, tuple))

    def expand(self, obj):
        return list(map(expand_json, obj))


@json_handler
class NumpyIntJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, np.integer)

    def expand(self, obj):
        return int(obj)


@json_handler
class NumpyFloatJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, np.floating)

    def expand(self, obj):
        return float(obj)


@json_handler
class NumpyArrayJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, np.ndarray)

    def expand(self, obj):
        return [expand_json(e) for e in obj.tolist()]


@json_handler
class PandasJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, (pd.DataFrame, pd.Series))

    def expand(self, obj):
        return obj.to_dict()


@json_handler
class IntPrimitiveClassJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, type) and obj == int

    def expand(self, obj):
        return 'int'


@json_handler
class FloatPrimitiveClassJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, type) and obj == float

    def expand(self, obj):
        return 'float'


@json_handler
class StringPrimitiveClassJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, type) and obj == str

    def expand(self, obj):
        return 'string'


@json_handler
class BoolPrimitiveClassJsonExpander(JsonExpander):
    def can_handle(self, obj):
        return isinstance(obj, type) and obj == bool

    def expand(self, obj):
        return 'boolean'


def expand_json(obj):
    for handler in reversed(json_handlers):
        if handler.can_handle(obj):
            return handler.expand(obj)
    return obj
