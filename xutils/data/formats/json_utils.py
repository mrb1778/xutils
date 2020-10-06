import json


def read(json_string):
    return json.loads(json_string)


def read_from(reader):
    return json.load(reader)


def write(obj, pretty_print=False, indent_amount=2):
    indent = indent_amount if pretty_print else None
    return json.dumps(obj, indent=indent)


def write_to(obj, writer, pretty_print=False, indent_amount=2):
    indent = indent_amount if pretty_print else None
    return json.dump(obj, writer, indent=indent)
