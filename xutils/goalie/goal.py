from __future__ import annotations
import argparse
import functools
import time
import re
import typing
from typing import Union, Dict, List, Callable, Literal, Optional, Tuple, Any, Type, Iterable, Sequence

import xutils.core.python_utils as pyu
import xutils.data.config_utils as cu
import xutils.data.json_utils as ju


class GoalDefinition:
    def __init__(self,
                 name: str,
                 scope: Optional[str] = None,
                 pre=None,
                 post=None,
                 params_as_goals=None,
                 params: Dict[str, GoalParam] = None,
                 refs=None,
                 executor: Callable = None,
                 executor_return_type: Optional[Type] = None,
                 executor_params: Iterable[Dict[str, Any]] = None,
                 goal_manager: GoalManager = None) -> None:
        super().__init__()

        self.goal_manager: GoalManager = goal_manager

        self.scope: str = scope
        self.name: str = name

        self.pre_goals: List[GoalDefinition] = list(goal_manager.find(scope=scope, goals=pre).values()) \
            if pre is not None else []
        self.post_goals: List[GoalDefinition] = list(goal_manager.find(scope=scope, goals=post).values()) \
            if post is not None else []
        self.ref_goals: List[GoalDefinition] = list(goal_manager.find(scope=scope, goals=refs).values()) \
            if refs is not None else []

        self.params: Dict[str, GoalParam] = params if params is not None else {}
        if params_as_goals is not None:
            for param_name, param_value in params_as_goals.items():
                if isinstance(param_value, GoalParam):
                    if param_value.name is None:
                        param_value.name = param_name
                    self.params[param_name] = param_value
                else:
                    self.params[param_name] = GoalParam(name=param_name,
                                                        goal=param_value)

        self.executor = executor
        self.executor_return_type: Optional[Type] = executor_return_type

        if self.executor is not None:
            functools.update_wrapper(self, self.executor)

        if executor_params is not None:
            for param in executor_params:
                param_name = param["name"]

                if param_name not in self.params:
                    default_value = param["default"]
                    if isinstance(default_value, GoalParam):
                        default_value.name = param_name
                        self.params[param_name] = default_value
                    else:
                        self.params[param_name] = GoalParam(name=param_name,
                                                            required=param["required"],
                                                            default=default_value,
                                                            data_type=param["type"],
                                                            kwargs=param["kwargs"])

    def execute(self, possible_kwargs):
        run_kwargs = {**self.param_defaults}
        for kwarg_name, kwarg_value in possible_kwargs.items():
            if kwarg_name in self.params:
                run_kwargs[kwarg_name] = kwarg_value

        return self.executor(**run_kwargs)

    def from_json(self, json):
        pass

    def referenced(self):
        return {
            *self.pre_goals,
            *self.post_goals,
            *self.ref_goals,
            *[param.goal for param in self.params_with_goals().values()]
        }

    def inputs(self) -> Dict[str, GoalParam]:
        found_inputs = {}

        for pre in self.pre_goals:
            if pre not in found_inputs:
                found_inputs.update(pre.inputs())

        for post in self.post_goals:
            if post not in found_inputs:
                found_inputs.update(post.inputs())

        for ref in self.ref_goals:
            if ref not in found_inputs:
                found_inputs.update(ref.inputs())

        for goal_param in self.params.values():
            found_inputs.update(goal_param.inputs())  # todo: circular references?

        return found_inputs

    def params_with_goals(self):
        return {param_name: param_value
                for param_name, param_value in self.params.items()
                if param_value.goal is not None}

    def param_kwargs(self) -> Optional[GoalParam]:
        for param in self.params.values():
            if param.kwargs:
                return param

        return None

    @property
    def param_defaults(self):
        """
        Returns a dictionary of parameter names and their default values.
        """
        return {param_name: param_value.default
                for param_name, param_value in self.params.items()
                if param_value.default is not None}

    def to_json(self):
        json = {
            "name": self.name
        }
        if len(self.pre_goals):
            json["pre"] = self.pre_goals

        if len(self.post_goals):
            json["post"] = self.post_goals

        if len(self.params):
            json["params"] = [
                p.to_json() for p in self.params.values()
            ]
        return json

    def curry(self, **kwargs):
        return GoalParam(goal=self,
                         run_goal=False,
                         **kwargs)

    def result(self, **kwargs):
        return GoalParam(goal=self,
                         run_goal=True,
                         **kwargs)

    def __call__(self, **kwargs):
        return self.run(**kwargs)

    def run(self, **kwargs):
        return self.goal_manager.run(self, **kwargs)

    def run_later(self, **kwargs):
        return self.goal_manager.run_later(self, **kwargs)

    def __str__(self) -> str:
        return F'{self.scope}:{self.name}' \
            if self.scope is not None else self.name


class CurriedGoal(GoalDefinition):
    def __init__(self,
                 goal: GoalDefinition = None,
                 fun_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs) -> None:
        super().__init__(name=goal.name, scope=goal.scope, goal_manager=goal.goal_manager)
        self.goal = goal
        self.fun_kwargs = fun_kwargs
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        return self.goal(**{**self.fun_kwargs, **self.kwargs, **kwargs})


class GoalRef(typing.NamedTuple):
    goal: str
    kwargs: Dict[str, Any]
    now: bool = False


class DynamicGoal(GoalDefinition):
    PARAMS_KEY = "params"

    def __init__(self,
                 name: str,
                 scope: Optional[str] = None,
                 goal_manager: GoalManager = None,
                 exec_goal: Union[str, GoalDefinition] = None,
                 goal_def: Union[str, Dict[str, Any]] = None,
                 run_regex: str = r'=(.*)',
                 run_param_regex: str = r'=\{\{(.*)\}\}',
                 ref_regex: str = r'\&(.*)',
                 param_regex: str = r'\{\{(.*)\}\}',
                 meta_param: str = '@') -> None:
        super().__init__(name=name, scope=scope, goal_manager=goal_manager)
        self.run_regex: re.Pattern = re.compile(run_regex)
        self.run_param_regex: re.Pattern = re.compile(run_param_regex)
        self.ref_regex: re.Pattern = re.compile(ref_regex)
        self.param_regex: re.Pattern = re.compile(param_regex)
        self.meta_param: str = meta_param
        # self.meta_regex: re.Pattern = re.compile(meta_regex)
        self.exec_goal = exec_goal
        self.goal_def = goal_def
        self._parse_meta()
        # self._populate_params()
        # self.params = {'kwargs': GoalParam(name='kwargs', kwargs=True)}
        # self.params = {}
        # self.params = {'kwargs': GoalParam(name='kwargs', kwargs=True)}

    def _parse_meta(self):
        # todo: values should be eval at run time not construction
        if isinstance(self.goal_def, dict):
            if len(self.goal_def) == 2 and self.meta_param in self.goal_def:
                self.meta_params = self.goal_def.get(self.meta_param)
                self.goal_def = self.goal_def.copy()
                self.goal_def.pop(self.meta_param)

                if self.PARAMS_KEY in self.meta_params:
                    # todo: implement
                    pass

    def _populate_params(self):
        self._parse_params(self.goal_def)

    def _parse_params(self, what):
        if isinstance(what, GoalDefinition):
            self.params.update(what.params)
        elif isinstance(what, str):
            # run_regex_check = re.fullmatch(self.run_regex, what)
            # ref_regex_check = re.fullmatch(self.ref_regex, what)
            run_param_regex_check = re.fullmatch(self.run_param_regex, what)
            param_regex_check = re.fullmatch(self.param_regex, what)

            # if ref_regex_check is not None:
            #     self.params.update(self.goal_manager.get(ref_regex_check.group(1)).params)
            # elif run_regex_check is not None:
            #     self.params.update(self.goal_manager.get(run_regex_check.group(1)).params)
            # elif param_regex_check is not None:
            if run_param_regex_check is not None:
                self.params[run_param_regex_check.group(1)] = GoalParam(name=run_param_regex_check.group(1),
                                                                        required=False)
            if param_regex_check is not None:
                self.params[param_regex_check.group(1)] = GoalParam(name=param_regex_check.group(1),
                                                                    required=False)
        elif isinstance(what, dict):
            first_key = next(iter(what))
            run_regex_check = re.fullmatch(self.run_regex, first_key)
            ref_regex_check = re.fullmatch(self.ref_regex, first_key)
            if len(what) == 1 and (run_regex_check is not None or ref_regex_check is not None):
                for goal_name, goal_params in what.items():
                    for param_name, param_value in goal_params.items():
                        self._populate_params(param_name)
                        self._populate_params(param_value)
                    if run_regex_check is not None:
                        self.params.update(self.goal_manager.get(run_regex_check.group(1)).params)
                    else:
                        self.params.update(self.goal_manager.get(ref_regex_check.group(1)).params)
            else:
                for key, value in what.items():
                    self._populate_params(key)
                    self._populate_params(value)
        elif isinstance(what, List):
            for item in what:
                self._populate_params(item)

    def execute(self, possible_kwargs: Dict[str, Any]):
        print("goal.py::execute:268:", possible_kwargs, ":(possible_kwargs)")
        dynamic_result = self._dynamic_execute(self.goal_def, possible_kwargs)
        if self.exec_goal is not None:
            print("goal.py::execute:270:", self.exec_goal, ":(self.exec_goal)")
            print("goal.py::execute:271:", dynamic_result, ":(dynamic_result)")
            return self.goal_manager.run(goal=self.exec_goal, possible_kwargs={**dynamic_result, **possible_kwargs})
        else:
            return dynamic_result

    def _dynamic_execute(self, what, possible_kwargs: Dict[str, Any]):
        # todo: move to @functools.singledispatchmethod, bugs with List Dict
        if isinstance(what, str):
            return self._dynamic_execute_str(what, possible_kwargs)
        elif isinstance(what, dict):
            return self._dynamic_execute_dict(what, possible_kwargs)
        elif isinstance(what, list):
            return self._dynamic_execute_list(what, possible_kwargs)
        else:
            return what

    def _dynamic_execute_str(self, what: str, possible_kwargs: Dict[str, Any]):
        run_regex_check = re.fullmatch(self.run_regex, what)
        ref_regex_check = re.fullmatch(self.ref_regex, what)
        param_regex_check = re.fullmatch(self.param_regex, what)

        if ref_regex_check is not None:
            # return self.goal_manager.run_later(ref_regex_check.group(1), fun_kwargs=possible_kwargs)
            return self.goal_manager.run_later(
                goal=DynamicGoal(name=f"{self.name}:RunLater:{ref_regex_check.group(1)}",
                                 goal_def=ref_regex_check.group(1),
                                 goal_manager=self.goal_manager),
                fun_kwargs=possible_kwargs.copy())
        elif run_regex_check is not None:
            run_param_regex_check = re.fullmatch(self.run_param_regex, what)
            if run_param_regex_check is not None:
                run_goal = possible_kwargs[run_param_regex_check.group(1)]
            else:
                run_goal = run_regex_check.group(1)
            print("goal.py::_dynamic_execute_str:303:", run_goal, ":(run_goal)")
            return self.goal_manager.run(run_goal, fun_kwargs=possible_kwargs)
        elif param_regex_check is not None:
            var_name = param_regex_check.group(1)
            if var_name not in possible_kwargs:
                return None
                # todo: only throw for execute, not ref
                # raise ValueError(
                #     f"Goal '{self.name}' expected '{var_name}'. Not found in: '[{', '.join(possible_kwargs.keys())}]'")
            else:
                return possible_kwargs[var_name]
        else:
            return what

    def _dynamic_execute_dict(self, what: Dict, possible_kwargs: Dict[str, Any]):
        dynamic_goal = self._get_dynamic_goal(what)

        if dynamic_goal is None:
            return {
                self._dynamic_execute(key, possible_kwargs=possible_kwargs):
                    self._dynamic_execute(value, possible_kwargs=possible_kwargs)
                for key, value in what.items()
            }
        elif dynamic_goal.now:
            print("goal.py::_dynamic_execute_dict:336:", possible_kwargs, ":(possible_kwargs)")
            return self.goal_manager.run(
                goal=dynamic_goal.goal,
                fun_kwargs=self._dynamic_execute(
                    what=dynamic_goal.kwargs,
                    possible_kwargs=possible_kwargs.copy()),
                **possible_kwargs.copy())
        else:
            dynamic_goal = DynamicGoal(
                name=f"{self.name}:{dynamic_goal.goal}:runLater",
                goal_def={f"={dynamic_goal.goal}": dynamic_goal.kwargs},
                goal_manager=self.goal_manager
            )
            dynamic_goal.run_later(**possible_kwargs)

    def _get_dynamic_goal(self, what: Dict) -> Optional[GoalRef]:
        keys = list(what.keys())
        if len(what) in (1, 2):
            first_key = keys[0]
            if len(what) == 1:
                fn_name = first_key
            elif len(what) == 2 and first_key == self.meta_param:
                fn_name = keys[1]
            else:
                return None

            fn_body = what[fn_name]

            run_match = re.fullmatch(self.run_regex, fn_name)
            ref_match = re.fullmatch(self.ref_regex, fn_name)
            if run_match is not None:
                return GoalRef(
                    now=True,
                    goal=run_match.group(1),
                    kwargs=fn_body
                )
            elif ref_match is not None:
                return GoalRef(
                    now=False,
                    goal=ref_match.group(1),
                    kwargs=fn_body
                )
            else:
                return None
        else:
            return None

    def _dynamic_execute_list(self, what: List, possible_kwargs: Dict[str, Any]):
        return [
            self._dynamic_execute(item, possible_kwargs=possible_kwargs)
            for item in what
        ]


class GoalParam:
    def __init__(self,
                 name: Optional[str] = None,
                 default: Optional[Any] = None,
                 label: Optional[str] = None,
                 description: Optional[str] = None,
                 data_type=None,
                 required: bool = False,
                 goal: Optional[Union[str, Callable, GoalDefinition]] = None,
                 run_goal: bool = False,
                 kwargs: bool = False,
                 **goal_kwargs) -> None:
        self.name = name

        self.default = default
        self.label = label
        self.description: str = description
        self.required: bool = required

        self.goal = goal
        self.run_goal: bool = run_goal if self.goal else False
        self.goal_inputs: Dict[str, GoalParam] = goal_kwargs

        if self.goal is None and isinstance(self.default, GoalDefinition):
            self.goal = self.default

        if data_type is not None:
            self.data_type = data_type
        elif self.goal is not None:
            if self.run_goal:
                self.data_type = self.goal.executor_return_type
            else:
                self.data_type = Callable
        elif self.default is not None:
            self.data_type = type(self.default)
        else:
            self.data_type = None

        self.kwargs = kwargs

    def inputs(self) -> Dict[str, GoalParam]:
        if self.goal:
            return {param_name: goal_param
                    for param_name, goal_param in self.goal.inputs().items() if param_name not in self.goal_inputs}
        else:
            return {self.name: self}

    def to_json(self):
        json = {
            "name": self.name
        }
        if self.default is not None:
            json["default"] = self.default
        if self.label is not None:
            json["label"] = self.label
        if self.description is not None:
            json["description"] = self.label
        if self.data_type is not None:
            json["dataType"] = self.data_type
        if self.required is not None:
            json["required"] = self.required

        if self.goal is not None:
            json["goal"] = self.goal
        if self.goal_inputs is not None and len(self.goal_inputs):
            json["goalKwargs"] = self.goal_inputs

        return json


class MissingGoal(Exception):
    def __init__(self, goal, scope=None, message: str = None) -> None:
        super().__init__(goal, scope, message)


class DuplicateGoal(Exception):
    def __init__(self, goal, scope=None, message: str = None) -> None:
        super().__init__(goal, scope, message)


class UnmetGoal(Exception):
    def __init__(self, goal, phase: Literal["run", "pre", "param", "post"], message: str) -> None:
        super().__init__(goal, phase, message)
        self.goal = goal
        self.phase = phase
        self.message = message

    def __str__(self) -> str:
        str_value = f"UnmetGoal({self.phase}:{self.goal})\n-->{self.__cause__.__str__()}\n"
        if isinstance(self.__cause__, self.__class__):
            str_value += self.message
        return str_value


class GoalScope:
    def __init__(self,
                 goal_manager: GoalManager,
                 scope: str):
        self.goal_manager: GoalManager = goal_manager
        self.previous_scope: Optional[str] = None
        self.scope: str = scope

    def __enter__(self):
        self.previous_scope = self.goal_manager.scope_default
        self.goal_manager.scope_default = self.scope
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.goal_manager.scope_default = self.previous_scope


class GoalRunContext:
    def __init__(self,
                 goal: Optional[str] = None,
                 scope: Optional[str] = None,
                 phase: Optional[str] = None,
                 parent: Optional[GoalRunContext] = None,
                 fun_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.goal = goal
        self.scope = scope
        self.phase = phase

        if parent is None:
            self.level: int = 0
        else:
            self.parent: GoalRunContext = parent
            self.level = self.parent.level + 1

        self.fun_kwargs = fun_kwargs if fun_kwargs is not None else dict()

        self.start_time = time.time()


class GoalManager:
    SCOPE_ALL = "__all__"
    SCOPE_STANDARD = "std"

    def __init__(self) -> None:
        self.print_logs = False
        self.lib: pyu.DictAccess[str, pyu.DictAccess[str, GoalDefinition]] = pyu.DictAccess()
        self.core_lib = {self.SCOPE_STANDARD: self.add_scope(self.SCOPE_STANDARD)}
        self.scope_default: Optional[str] = None

    def get_scope(self, scope: Optional[str] = None):
        return self.scope_default if scope is None else scope

    def add_scope(self,
                  scope: str,
                  values: Optional[pyu.DictAccess[str, GoalDefinition]] = None) -> pyu.DictAccess[str, GoalDefinition]:
        if scope not in self.lib:
            scoped_goals = pyu.DictAccess() if values is None else values
            self.lib[scope] = scoped_goals
            return scoped_goals
        else:
            return self.lib[scope]

    def in_scope(self, scope: str):
        return GoalScope(goal_manager=self, scope=scope)

    def find(self,
             scope: Optional[str] = None,
             goals: Optional[List[Union[str, Callable, GoalDefinition]]] = None,
             search_core: bool = True,
             fail_if_missing: bool = True) -> pyu.DictAccess[str, GoalDefinition]:
        scope = self.get_scope(scope)

        if goals is not None:
            if not isinstance(goals, List):
                goals = [goals]
            if len(goals) == 0:
                return pyu.DictAccess()

        scoped_goals = self.lib.get(scope)
        if search_core:
            core_lib = self.core_lib.get(self.SCOPE_STANDARD)
            if scoped_goals is None:
                scoped_goals = core_lib
            elif core_lib is not None:
                new_scoped_goals = pyu.DictAccess()
                new_scoped_goals.update(core_lib)
                new_scoped_goals.update(scoped_goals)
                scoped_goals = new_scoped_goals

        if scoped_goals is None:
            return pyu.DictAccess()
        elif goals is None:
            return scoped_goals
        else:
            filtered_goals = pyu.DictAccess()
            for goal in goals:
                if isinstance(goal, GoalDefinition):
                    filtered_goals[goal.name] = goal
                elif isinstance(goal, str):
                    if goal in scoped_goals:
                        filtered_goals[goal] = scoped_goals[goal]
                    elif fail_if_missing:
                        raise MissingGoal(goal, scope)
                else:
                    raise ValueError(f"{goal} is not a valid goal type")

            return filtered_goals

    def get(self, goal,
            scope: Optional[str] = None,
            fail_if_missing: bool = True) -> GoalDefinition:
        found_goals = self.find(scope=scope,
                                goals=[goal],
                                fail_if_missing=fail_if_missing)
        return next(iter(found_goals.values()))

    def goal_names(self, scope: Optional[str] = None) -> List[str]:
        return list(self.find(scope=scope).keys())

    def reset(self, scope: Optional[str] = None) -> None:
        self.scope_default = None
        scope = self.get_scope(scope)

        if scope == self.SCOPE_ALL or scope is None:
            self.lib.clear()
            for core_key, core_value in self.core_lib.items():
                self.add_scope(scope=core_key, values=core_value)
        elif scope in self.lib:
            self.lib[scope].clear()

    def add(self,
            goal_def: GoalDefinition,
            overwrite=False) -> GoalDefinition:
        scope = self.get_scope(goal_def.scope)
        scoped_goals = self.lib.get(scope)
        if scoped_goals is None:
            scoped_goals = self.add_scope(scope=scope)
        if not overwrite and goal_def.name in scoped_goals:
            raise DuplicateGoal(goal_def.name, scope)
        scoped_goals[goal_def.name] = goal_def

        goal_def.goal_manager = self

        return goal_def

    def __call__(self,
                 function: Optional[Callable] = None,
                 name: Optional[str] = None,
                 pre=None,
                 post=None,
                 params=None,
                 refs=None,
                 scope: Optional[str] = None) -> Callable:

        scope = self.get_scope(scope)

        def decorator_goal(fun: Callable):
            nonlocal name
            if name is None:
                name = fun.__name__

            goal_def = GoalDefinition(name=name,
                                      scope=scope,
                                      executor=fun,
                                      executor_return_type=pyu.return_type(fun),
                                      executor_params=pyu.params(fun).values(),
                                      pre=pre,
                                      post=post,
                                      params_as_goals=params,
                                      refs=refs,
                                      goal_manager=self)
            self.add(goal_def)
            return goal_def

        if function:
            return decorator_goal(function)
        return decorator_goal

    # noinspection PyMethodMayBeStatic
    # def param(self,
    #           name: Optional[Union[str, Callable, GoalDefinition]] = None,
    #           source=None,
    #           default=None,
    #           label=None,
    #           description=None,
    #           data_type=None,
    #           required: bool = False,
    #           goal_id: Optional[Union[str, Callable, GoalDefinition]] = None,
    #           scope: Optional[str] = None,
    #           **kwargs):
    #     return GoalParam(name=name,
    #                      source=source,
    #                      default=default,
    #                      label=label,
    #                      description=description,
    #                      data_type=data_type,
    #                      required=required,
    #                      goal=goal_id,
    #                      scope=scope,
    #                      **kwargs)

    # def leafs(self, scope: Optional[str] = None):
    #     #  todo: optimize and return 1st one instead
    #     all_goals = self.find(scope=scope)
    #     leafs = []
    #     for goal_name in all_goals.keys():
    #         if len(self.requirements(goal_name, scope=scope)) == 0:
    #             leafs.append(goal_name)
    #
    #     return leafs
    #
    # def roots(self, scope: Optional[str] = None):
    #     #  todo: optimize
    #     all_goals = self.goal_names(scope=scope)
    #     all_requirements = {}
    #
    #     for goal_name in all_goals:
    #         all_goals.extend(self.requirements(goal_name, scope=scope))
    #
    #     return [goal_name for goal_name in all_goals if goal_name not in all_requirements]

    def inputs(self,
               goal: Optional[Any[str, GoalDefinition]] = None,
               scope: Optional[str] = None,
               only_required: bool = False) -> Dict[str, GoalParam]:
        scope = self.get_scope(scope)

        if goal is None:
            inputs = {}
            for goal in self.find(scope=scope).values():
                inputs.update(goal.inputs())
        else:
            goal = self.get(goal, scope=scope)
            inputs = goal.inputs()

        if only_required:
            return {key: goal_param for key, goal_param in inputs.items() if goal_param.required}
        else:
            return inputs

    def run(self,
            goal: Union[str, GoalDefinition] = None,
            scope: Optional[str] = None,
            context: Optional[GoalRunContext] = None,
            fun_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs):
        scope = self.get_scope(scope)

        context = GoalRunContext(goal=goal,
                                 scope=scope,
                                 parent=context,
                                 phase="Run")

        fun_kwargs = {
            **(fun_kwargs if fun_kwargs is not None else {}),
            **kwargs
        }

        # self.log(context, "Start", fun_kwargs)
        self.log(context, "Start")

        goal_def = self.get(goal, scope=scope)
        if goal_def is None:
            raise UnmetGoal(goal,
                            phase="run",
                            message=f"Error in run goal is missing")

        updated_kwargs = fun_kwargs.copy()

        results = []
        for pre in goal_def.pre_goals:
            try:
                result = self.run(context=context,
                                  goal=pre,
                                  scope=scope,
                                  fun_kwargs=fun_kwargs)
                results.append(result)
            except Exception as e:
                raise UnmetGoal(goal=pre, phase="pre", message="Pre Failed") from e

        for param_name, param_value in goal_def.params_with_goals().items():
            param_goal: GoalDefinition = param_value.goal
            try:
                if param_value.run_goal:
                    print("goal.py::run:766")
                    updated_kwargs[param_name] = self.run(context=context,
                                                          scope=scope,
                                                          goal=param_goal,
                                                          fun_kwargs=param_value.goal_inputs,
                                                          **updated_kwargs)
                else:
                    print("goal.py::run:773")
                    updated_kwargs[param_name] = self.run_later(goal=param_goal,
                                                                scope=scope,
                                                                fun_kwargs=param_value.goal_inputs,
                                                                **updated_kwargs)
            except Exception as e:
                print('Unmet', {**param_value.goal_inputs,
                                **updated_kwargs})
                raise UnmetGoal(goal,
                                phase="param",
                                message=f"Error in param {goal}({param_name}:{param_goal.name}) -> {e}") from e
        self.log(context, "Run")

        result = goal_def.execute(updated_kwargs)

        for post in goal_def.post_goals:
            try:
                result = self.run(context=context,
                                  goal=post,
                                  scope=scope,
                                  fun_kwargs=fun_kwargs)
                results.append(result)
            except Exception as e:
                raise UnmetGoal(goal=goal, phase="post", message="Error in post") from e

        self.log(context, "End")
        return result

        # updated_kwargs[param_name] = lambda **param_invoke_kwargs: self._run(
        #     context=context,
        #     goal=param_goal,
        #     scope=scope,
        #     fun_kwargs={**param_value.goal_inputs,
        #                 **updated_kwargs,
        #                 **param_invoke_kwargs})

    def run_later(self,
                  goal: Union[str, GoalDefinition] = None,
                  scope: Optional[str] = None,
                  fun_kwargs: Optional[Dict[str, Any]] = None,
                  **kwargs):
        scope = self.get_scope(scope)
        goal_def = self.get(goal, scope=scope)

        return CurriedGoal(goal=goal_def,
                           fun_kwargs=fun_kwargs if fun_kwargs is not None else {},
                           **kwargs)

    def run_batch(self,
                  batch=None,
                  path=None,
                  scope: Optional[str] = None,
                  run_kwargs=None):
        scope = self.get_scope(scope)

        if run_kwargs is None:
            run_kwargs = {}

        if path is not None:
            batch = cu.read_file(path)
        elif isinstance(batch, str):
            batch = ju.read(batch)

        if batch is None:
            raise ValueError("json string or file required")

        if "debug" in batch and not self.print_logs:
            self.debug(batch["debug"])

        if "params" in batch:
            run_kwargs.update(batch["params"])

        if "goals" in batch:
            goals = batch["goals"]
            for goal_name, goal_def in goals.items():
                self.add(DynamicGoal(name=goal_name,
                                     goal_def=goal_def,
                                     goal_manager=self))

        if "run" in batch:
            run = batch["run"]
            return self.run(DynamicGoal(name=":BatchGoal:",
                                        goal_def=run,
                                        goal_manager=self),
                            fun_kwargs=run_kwargs,
                            scope=scope)

    def all_goals(self):
        return {scope: goals.keys() for scope, goals in self.lib.items()}

    def to_json(self):
        return {
            "goals": {key: goals for key, goals in self.lib.items()}
        }

    def export_goals(self, file_name: str):
        ju.write_to(self, path=file_name)

    def create_arg_parser(self,
                          default_goal=None,
                          all_args=True,
                          scope: Optional[str] = None,
                          custom_inputs: Dict[str, GoalParam] = None,
                          **kwargs):
        scope = self.get_scope(scope)

        parser = argparse.ArgumentParser()
        parser.add_argument("--goal", "--g",
                            type=str,
                            default=default_goal,
                            help="Which goal to execute (default: %(default)s). Options: " +
                                 ', '.join(self.goal_names(scope=scope)))
        parser.add_argument('--inspect',
                            dest='inspect',
                            action='store_true',
                            help='Inspect Goal')
        parser.set_defaults(inspect=False)

        parser.add_argument('--debug',
                            dest='debug',
                            action='store_true',
                            help='Inspect Goal')
        parser.set_defaults(debug=False)

        parser.add_argument("--goal_config",
                            type=str,
                            help="Goal JSON/Yaml file, command line params will override")
        self._create_arg_parser_params(arg_parser=parser,
                                       default_goal=default_goal,
                                       all_args=all_args,
                                       scope=scope,
                                       custom_inputs=custom_inputs,
                                       **kwargs)
        return parser

    def _create_arg_parser_params(self,
                                  arg_parser: argparse.ArgumentParser,
                                  default_goal,
                                  scope: Optional[str] = None,
                                  custom_inputs: Dict[str, GoalParam] = None,
                                  enforced_required=False,
                                  all_args=True,
                                  **kwargs):
        scope = self.get_scope(scope)

        # all_goals = self.goal_names(scope=scope)
        # root_goals = self.roots(scope=scope)
        # for root_goal in root_goals:
        #     del all_goals[root_goal]
        #     all_goals.append(f"{root_goal}*")

        goal_args_group = arg_parser.add_argument_group("All Args" if all_args else f"{default_goal} Args")
        inputs = self.inputs(goal=None if all_args else default_goal, scope=scope)
        for input_name, goal_param in inputs.items():
            if custom_inputs is not None and input_name in custom_inputs:
                goal_input = custom_inputs[input_name]

            input_default = kwargs[input_name] if input_name in kwargs else goal_param.default

            help_text = ""
            if goal_param.description is not None:
                help_text = goal_param.description
            if input_default is not None:
                help_text += " (default: %(default)s)"

            # noinspection PyTypeChecker
            goal_args_group.add_argument(f"--{input_name}",
                                         default=input_default,
                                         type=goal_param.data_type if goal_param.data_type != List else None,
                                         # required=True if input_default is None else False,
                                         required=goal_param.required if enforced_required else False,
                                         nargs="*" if goal_param.data_type == List else "?",
                                         action="store",
                                         help=help_text)

    def run_from_arg_parse(self,
                           arg_parser: argparse.ArgumentParser = None,
                           default_goal=None,
                           all_args: bool = True,
                           scope: Optional[str] = None,
                           custom_inputs: Optional[Dict[str, GoalParam]] = None,
                           print_result: bool = True,
                           **kwargs):
        scope = self.get_scope(scope)

        if arg_parser is None:
            arg_parser = self.create_arg_parser(default_goal=default_goal,
                                                all_args=all_args,
                                                scope=scope,
                                                custom_inputs=custom_inputs,
                                                **kwargs)
        args = pyu.parse_unknown_args(arg_parser)
        args = pyu.remove_na_from_dict(args)
        # for goal_name in goal_names:
        #     self._create_arg_parser_params(arg_parser=arg_parser,
        #                                    default_goal=goal_name,
        #                                    scope=scope,
        #                                    custom_inputs=custom_inputs,
        #                                    enforced_required=True)
        # args = arg_parser.parse_args()

        result = None
        if "debug" in args:
            if args["debug"]:
                self.debug(args["debug"])
        if "goal_config" in args:
            goal_config = args["goal_config"]
            del args['goal_config']
            result = self.run_batch(path=goal_config, scope=scope, run_kwargs={**kwargs, **args})
        elif "goal" in args:
            goal_name = args["goal"]
            # goal_names = goal_name.split(" ")
            del args['goal']

            if args["inspect"]:
                print(*(next(iter(self.find(goals=[goal_name], scope=scope).values())).referenced()))
            else:
                result = self.run(goal_name, scope=scope, **{**kwargs, **args})
        else:
            raise ValueError("no goal or json specified")

        if print_result and result is not None:
            print(result)

        return result

    def log(self, context, action, *args):
        if self.print_logs:
            pyu.print_bordered(*(context.level * '-'),
                               f"Goal: {context.goal}",
                               *((context.scope, "::") if context.scope is not None else ()),
                               action,
                               f"Time: {time.time() - context.start_time}",
                               *args)

    def debug(self, on: bool = True) -> None:
        self.print_logs = on

# todo: bring back for default scope
# class GoalScope:
#     def __init__(self, goal_registry: GoalRegistry, scope: str):
#         self.goal_registry = goal_registry
#         self.scope = scope
#         self.previous_scope = None
#
#     def __enter__(self):
#         self.activate()
#
#     def activate(self):
#         self.previous_scope = self.goal_registry.current_scope
#         self.goal_registry.current_scope = self
#
#     def __exit__(self, *args):
#         self.deactivate()
#
#     def deactivate(self):
#         self.goal_registry.current_scope = self.previous_scope
