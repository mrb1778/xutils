import functools
import importlib
import itertools
import os.path
from typing import Any, Callable, List, Iterable, Dict, Union, Optional

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
    def _print(what: str = "hello") -> None:
        if isinstance(what, pd.DataFrame):
            pu.print_all(what)
        elif isinstance(what, Dict):
            pyu.print_dict(what)
        else:
            print(what)

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
    def _get(obj, what: str):
        return getattr(obj, what)

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
    def _multiply(x: Any, y: Any) -> Any:
        return x * y

    @goal(name="divide", scope=scope)
    def _divide(x: Any, y: Any) -> Any:
        return x / y

    @goal(name="add", scope=scope)
    def _add(x: Any, y: Any) -> Any:
        return x + y

    @goal(name="subtract", scope=scope)
    def _subtract(x: Any, y: Any) -> Any:
        return x - y

    @goal(name="with", scope=scope)
    def _with(do: Iterable[Callable], kwargs: Dict):
        print("std_lib.py::_with:106:", kwargs, ":(kwargs)", do)
        last_result = None
        for do_item in do:
            last_result = do_item(**kwargs)
        return last_result

    @goal(scope=scope)
    def collect(over: Union[Iterable[Any], Dict[str, Iterable[Any]]],
                do: Union[Any, Callable[[Any], Any]],
                loop_item: Optional[str] = None,
                as_kwargs: bool = False) -> List:
        if isinstance(do, str):
            do = goal.get(do)
        # todo: how to handle non permute
        if isinstance(over, Dict):
            merged_args = pyu.permute_transpose(over)
            return [do(**fun_kwargs) for fun_kwargs in merged_args]
        else:
            return [do(**e) if as_kwargs else do(e) if loop_item is None
                    else do(**{loop_item: e}) for e in over]

    @goal(scope=scope)
    def transpose(what: Dict[str, Iterable[Any]]) -> List:
        return pyu.permute_transpose(what)


    @goal(scope=scope)
    def loop(over: Union[Iterable[Any], Dict[str, Iterable[Any]]],
             do: Union[Any, Callable[[Any], Any]],
             loop_item: Optional[str] = None) -> None:
        if isinstance(do, str):
            do = goal.get(do)
        # todo: how to handle non permute
        if isinstance(over, Dict):
            merged_args = pyu.permute_transpose(over)
            for fun_kwargs in merged_args:
                print("std_lib.py::loop:136:",  fun_kwargs, ":(fun_kwargs)")
                do(**fun_kwargs)
        else:
            for e in over:
                if loop_item:
                    do(**{loop_item: e})
                else:
                    do(e)

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
    def _reduce(these: Iterable, how: Optional[Callable] = None, initial: Optional[Any] = None) -> object:
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
