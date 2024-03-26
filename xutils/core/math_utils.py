import statistics
from numbers import Number
from typing import List


def mean(*numbers: Number) -> float:
    return statistics.mean(numbers)