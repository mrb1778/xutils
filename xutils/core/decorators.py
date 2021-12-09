import re

class DecoratorException(Exception):
    headline = "Flow failed"

    def __init__(self, msg="", lineno=None):
        self.message = msg
        self.line_no = lineno
        super(DecoratorException, self).__init__()

    def __str__(self):
        prefix = f"line {self.line_no:d}: " if self.line_no else ""
        return "%s%s" % (prefix, self.message)


class InvalidDecoratorAttribute(DecoratorException):
    headline = "Unknown decorator attribute"

    def __init__(self, deconame, attr, defaults):
        msg = (
            "Decorator '{deco}' does not support the attribute '{attr}'. "
            "These attributes are supported: {defaults}.".format(
                deco=deconame, attr=attr, defaults=", ".join(defaults)
            )
        )
        super(InvalidDecoratorAttribute, self).__init__(msg)


class Decorator(object):
    """
    Base class for all decorators.
    """

    name = "NONAME"
    defaults = {}

    def __init__(self, attributes=None, statically_defined=False):
        self.attributes = self.defaults.copy()
        self.statically_defined = statically_defined

        if attributes:
            for k, v in attributes.items():
                if k in self.defaults:
                    self.attributes[k] = v
                else:
                    raise InvalidDecoratorAttribute(self.name, k, self.defaults)

    @classmethod
    def _parse_decorator_spec(cls, deco_spec):
        top = deco_spec.split(":", 1)
        if len(top) == 1:
            return cls()
        else:
            name, attrspec = top
            attrs = dict(
                map(lambda x: x.strip(), a.split("="))
                for a in re.split(""",(?=[\s\w]+=)""", attrspec.strip("\"'"))
            )
            return cls(attributes=attrs)

    def make_decorator_spec(self):
        attrs = {k: v for k, v in self.attributes.items() if v is not None}
        if attrs:
            attrstr = ",".join("%s=%s" % x for x in attrs.items())
            return "%s:%s" % (self.name, attrstr)
        else:
            return self.name

    def __str__(self):
        mode = "decorated" if self.statically_defined else "cli"
        attrs = " ".join("%s=%s" % x for x in self.attributes.items())
        if attrs:
            attrs = " " + attrs
        fmt = "%s<%s%s>" % (self.name, mode, attrs)
        return fmt