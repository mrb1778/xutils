import re

import numpy as np


def read_lines(path, strip=True):
    with open(path) as f:
        lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    return lines


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


_camel_snake_regex_name_letters = re.compile('(.)([A-Z][a-z]+)')
_camel_snake_regex_name_numbers = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(camel_str):
    camel_str = re.sub(_camel_snake_regex_name_letters, r'\1_\2', camel_str)
    return re.sub(_camel_snake_regex_name_numbers, r'\1_\2', camel_str).lower()


def snake_to_camel(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])
