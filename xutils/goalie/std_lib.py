import functools
import importlib
import itertools
import os.path
from typing import Any, Callable, List, Iterable, Dict, Union, Optional

import numpy as np
import pandas as pd

import xutils.core.python_utils as pyu
import xutils.data.pandas_utils as pu
from xutils.goalie import GoalManager


def init_lib(goal: GoalManager, scope: str):
    @goal(scope=scope)
    def hello_world() -> str:
        return "Hello Goal World!"

    @goal(scope=scope)
    def echo(value: str = "echo") -> str:
        return value

    @goal(name="print", scope=scope)
    def _print(what: str = "hello") -> Any:
        if isinstance(what, pd.DataFrame):
            return pu.print_all(what)
        elif isinstance(what, Dict):
            return pyu.print_dict(what)
        else:
            print(what)
            return what

    @goal(name="range", scope=scope)
    def _range(start: int = 0, stop: int = 10) -> range:
        return range(start, stop)

    @goal(name="exec", scope=scope)
    def _exec(fun: str,
              package: Optional[str] = None,
              obj: Optional[Any] = None,
              args: Optional[List] = None,
              kwargs: Optional[Dict] = None) -> Any:
        return pyu.execute(fun=fun, package=package, obj=obj, args=args, kwargs=kwargs)

    @goal(name="get", scope=scope)
    def _get(obj, what: Union[str, Iterable[str]]):
        if isinstance(what, str):
            return getattr(obj, what)
        else:
            return [getattr(obj, what_item) for what_item in what]

    @goal(name="set", scope=scope)
    def _set(obj: Any, what: str, value: Any) -> Any:
        obj[what] = value
        # setattr(obj, what, value)
        return obj

    @goal(name="index", scope=scope)
    def _index(obj, what: Any):
        return obj[what]

    @goal(name="if", scope=scope)
    def _if(test: Union[bool, Callable],
            then: Union[Any, Callable] = True,
            otherwise: Union[Any, Callable] = False) -> None:
        if isinstance(test, Callable):
            test = test()

        if test:
            return then() if isinstance(then, Callable) else then
        else:
            return otherwise() if isinstance(otherwise, Callable) else otherwise

    @goal(name="equals", scope=scope)
    def _equals(x: Any, y: Any) -> bool:
        return x == y

    @goal(name="gt", scope=scope)
    def gt(x: Any, y: Any) -> bool:
        return x > y

    @goal(name="gte", scope=scope)
    def gte(x: Any, y: Any) -> bool:
        return x >= y

    @goal(name="lt", scope=scope)
    def lt(x: Any, y: Any) -> bool:
        return x < y

    @goal(name="lte", scope=scope)
    def lte(x: Any, y: Any) -> bool:
        return x <= y

    @goal(name="and", scope=scope)
    def _and(x: Any, y: Any) -> bool:
        return x and y

    @goal(name="or", scope=scope)
    def _or(x: Any, y: Any) -> bool:
        return x or y

    @goal(name="multiply", scope=scope)
    def multiply(x: Any, y: Any) -> Any:
        return x * y

    @goal(name="divide", scope=scope)
    def divide(x: Any, y: Any) -> Any:
        return x / y

    @goal(name="add", scope=scope)
    def add(x: Any, y: Any) -> Any:
        return x + y

    @goal(name="subtract", scope=scope)
    def subtract(x: Any, y: Any) -> Any:
        return x - y

    @goal(name="avg", scope=scope)
    def avg(x: Iterable) -> Any:
        return np.average(np.array(x))

    @goal(name="with", scope=scope)
    def _with(do: Union[Callable, Dict[str, Callable], Iterable[Callable]],
              kwargs: Dict[str, Any]):
        if isinstance(do, Callable):
            return do(**kwargs)
        elif isinstance(do, Dict):
            result = None
            kwargs = kwargs.copy()
            for name, do_item in do.items():
                result = do_item(**kwargs)
                kwargs[name] = result
            return result
        elif isinstance(do, Iterable):
            last_result = None
            for do_item in do:
                last_result = do_item(**kwargs)
            return last_result
        else:
            raise ValueError(f"Invalid do {do}, must be Callable, Dict[str, Callable], or Iterable[Callable]")

    @goal(scope=scope)
    def loop(over: Union[Iterable[Any], Dict[str, Iterable[Any]]],
             do: Union[Callable[[Any], Any], Iterable[Callable]],
             loop_item: Optional[str] = None) -> Any:
        if isinstance(do, str):
            do = goal.get(do)
        # todo: how to handle non permute
        result = None
        if isinstance(over, Dict):
            merged_args = pyu.permute_transpose(over)
            for fun_kwargs in merged_args:
                # do(**fun_kwargs)
                result = _with(do=do, kwargs=fun_kwargs)
        else:
            print("std_lib.py::loop:158")
            for e in over:
                if loop_item:
                    # do(**{loop_item: e})
                    result = _with(do=do, kwargs={loop_item: e})
                else:
                    # do(e)
                    result = _with(do=do, kwargs=e)
        return result

    @goal(scope=scope)
    def collect(over: Union[Iterable[Any], Dict[str, Iterable[Any]]],
                do: Union[Any, Callable[[Any], Any]],
                loop_item: Optional[str] = None) -> List:
        if isinstance(do, str):
            do = goal.get(do)
        # todo: how to handle non permute
        if isinstance(over, Dict):
            merged_args = pyu.permute_transpose(over)
            return [_with(do=do, kwargs=fun_kwargs) for fun_kwargs in merged_args]
        else:
            return [_with(do=do, kwargs=e) if loop_item is None
                    else _with(do=do, kwargs={loop_item: e}) for e in over]

    @goal(scope=scope)
    def transpose(what: Dict[str, Iterable[Any]]) -> List:
        return pyu.permute_transpose(what)


    # @goal(scope=scope)
    # def query(from_: Dict[str, Any], select: Optional[Iterable[Any]] = None):
    #     return_val = {key, }

    # @goal(scope=scope)
    # def nested_loop(over: Dict[str, List[Any]], body: Callable) -> List:
    #     results = []
    #
    #     for index in range(max(map(len, over.values()))):
    #         args = {key: value[index] for key, value in over.items()}
    #         results.append(body(**args))
    #
    #     return results

    @goal(name="merge", scope=scope)
    def _merge(these: Iterable, how: Optional[Callable] = None, initial: Optional[Any] = None) -> object:
        return pu.merge_all(these)

    @goal(name="reduce", scope=scope)
    def reduce(these: Iterable, how: Optional[Callable] = None, initial: Optional[Any] = None) -> object:
        if how is not None:
            return functools.reduce(how, these, initial)
        else:
            items_iter = iter(these)
            first_item = next(items_iter)
            full_items = itertools.chain([first_item], items_iter)

            if isinstance(first_item, pd.DataFrame):
                concat_results = pu.concat_unique(full_items)
            elif isinstance(first_item, Dict):
                concat_results = pu.concat_dicts(full_items)
            else:
                concat_results = list(full_items)

            return concat_results

    @goal(scope=scope)
    def plus(x, y):
        return x + y

    @goal(scope=scope)
    def concat(x, y):
        return x + y

    # path
    @goal(scope=scope)
    def os_join(args: List[Any]) -> str:
        return os.path.join(*args)
