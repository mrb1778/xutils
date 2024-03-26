from __future__ import annotations

import functools
import time
import typing
from typing import Union, Dict, List, Callable, Literal, Optional, Any, Type, Iterable

import xutils.core.python_utils as pyu
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

        # if name == "my_fun_args":
        #     print("goal.py::__init__:27:", name)
        #     if executor_params is not None:
        #         for p in executor_params:
        #             pyu.print_dict(p)

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
                                                            positional=param["positional"],
                                                            args=param["args"],
                                                            kwargs=param["kwargs"])

        self.executor_has_args_param = any(param.args for param in self.params.values())
        self.executor_has_kwargs_param = any(param.kwargs for param in self.params.values())

    def execute(self, possible_kwargs):
        run_args = []
        run_kwargs = {}
        for name, param in self.params.items():
            if name in possible_kwargs:
                value = possible_kwargs[name]
                if self.executor_has_args_param:
                    if param.args:
                        run_args.extend(value)
                    elif param.required:
                        run_args.append(value)
                    else:
                        run_kwargs[name] = value
                else:
                    run_kwargs[name] = value
            elif param.default is not None:
                run_kwargs[name] = param.default
            # todo: required but not found?

        return self.executor(*run_args, **run_kwargs)

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

    def param_args(self) -> Optional[GoalParam]:
        for param in self.params.values():
            if param.args:
                return param

        return None

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


class GoalParam:
    def __init__(self,
                 name: Optional[str] = None,
                 default: Optional[Any] = None,
                 label: Optional[str] = None,
                 description: Optional[str] = None,
                 data_type=None,
                 positional: bool = False,
                 required: bool = False,
                 goal: Optional[Union[str, Callable, GoalDefinition]] = None,
                 run_goal: bool = False,
                 args: bool = False,
                 kwargs: bool = False,
                 **goal_kwargs) -> None:
        self.name = name

        self.default = default
        self.label = label
        self.description: str = description
        self.positional: bool = positional
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

        self.args = args
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

    def get(self,
            goal: Union[str, Callable, GoalDefinition],
            scope: Optional[str] = None,
            fail_if_missing: bool = True) -> GoalDefinition:
        # found_goals = self.find(scope=scope,
        #                         goals=[goal],
        #                         fail_if_missing=fail_if_missing)
        # return next(iter(found_goals.values()))
        if isinstance(goal, GoalDefinition):
            return goal
        elif isinstance(goal, str):
            if goal in self.lib.get(scope):
                return self.lib.get(scope)[goal]
            elif goal in self.core_lib.get(self.SCOPE_STANDARD):
                return self.core_lib.get(self.SCOPE_STANDARD)[goal]
            else:
                if "." in goal:
                    scope_goal = goal.rsplit('.', 1)
                    if len(scope_goal) == 2:
                        split_scope, split_goal = scope_goal
                        if split_goal in self.lib.get(split_scope):
                            return self.lib.get(split_scope)[split_goal]
        else:
            raise ValueError(f"{goal} is not a valid goal type")

        raise MissingGoal(goal, scope)

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

        goal_def = self.get(goal, scope=scope)
        if goal_def is None:
            raise UnmetGoal(goal,
                            phase="run",
                            message=f"Error in run goal is missing")

        scope = goal_def.scope

        context = GoalRunContext(goal=goal,
                                 scope=scope,
                                 parent=context,
                                 phase="Run")

        # self.log(context, "fun_kwargs", fun_kwargs)
        fun_kwargs = {
            **(fun_kwargs if fun_kwargs is not None else {}),
            **kwargs
        }

        # self.log(context, "Start", fun_kwargs)
        self.log(context, "Start")

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
                    updated_kwargs[param_name] = self.run(context=context,
                                                          scope=scope,
                                                          goal=param_goal,
                                                          fun_kwargs=param_value.goal_inputs,
                                                          **updated_kwargs)
                else:
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

    def all_goals(self):
        return {scope: goals.keys() for scope, goals in self.lib.items()}

    def to_json(self):
        return {
            "goals": {key: goals for key, goals in self.lib.items()}
        }

    def export_goals(self, file_name: str):
        ju.write_to(self, path=file_name)

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

    def add_package(self, package: str, scope: str = None) -> None:
        if scope is None:
            scope = package

        package_funs = pyu.get_functions(module=package)
        for name, fun in package_funs:
            goal_def = GoalDefinition(name=name,
                                      scope=scope,
                                      executor=fun,
                                      executor_return_type=pyu.return_type(fun),
                                      executor_params=pyu.params(fun).values(),
                                      goal_manager=self)
            self.add(goal_def)


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
