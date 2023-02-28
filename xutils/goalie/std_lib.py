import collections
from typing import Any, Callable, List

from xutils.goalie import GoalManager


def init_lib(goal: GoalManager, scope: str):
    @goal(scope=scope)
    def echo(value: str) -> str:
        return value

    @goal(scope=scope)
    def loop(over, body) -> List:
        results = []
        for e in over:
            results.append(body(e))

        return results

    @goal(scope=scope)
    def join(over: collections.Iterable = None, how: Callable = None, initial: Any = None) -> object:
        result = initial
        for e in over:
            result = how(result, e)

        return result


