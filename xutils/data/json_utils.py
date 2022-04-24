import inspect
import json

import numpy as np


def read(json_string):
    return json.loads(json_string)


def read_from(reader):
    return json.load(reader)


def read_file(path):
    with open(path) as file:
        return json.load(file)


def write(obj, pretty_print=False, indent_amount=2):
    indent = indent_amount if pretty_print else None
    return json.dumps(obj, indent=indent, cls=ObjectEncoder)


def write_to(obj, writer=None, path=None, pretty_print=False, indent_amount=2):
    indent = indent_amount if pretty_print else None
    if writer is None:
        with open(path, 'wt') as writer:
            json.dump(obj, writer, indent=indent, cls=ObjectEncoder)
            return path
    else:
        json.dump(obj, writer, indent=indent, cls=ObjectEncoder)
        return writer


# todo: make list of smart encoders


class ObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_json"):
            return self.default(obj.to_json())
        if isinstance(obj, (np.ndarray, np.int64, np.float32, np.float64)):
            return obj.tolist()
        elif hasattr(obj, "__dict__"):
            d = dict(
                (key, value)
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
            )
            return self.default(d)
        return json.JSONEncoder.default(self, obj)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.int64)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
