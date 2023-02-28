from __future__ import annotations
import argparse
import functools
import time
from typing import Union, Dict, List, Callable, Literal, Optional, Tuple, Any, Set, Type, Iterable

import xutils.core.python_utils as pu
import xutils.data.json_utils as ju


class GoalDefinition:
    def __init__(self,
                 executor: Callable,
                 name: str,
                 scope: Optional[str] = None,
                 pre=None,
                 post=None,
                 params=None,
                 refs=None,
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

        goal_params = {}
        if params is not None:
            for param_name, param_value in params.items():
                if isinstance(param_value, GoalParam):
                    if param_value.name is None:
                        param_value.name = param_name
                    goal_params[param_name] = param_value
                else:
                    goal_params[param_name] = GoalParam(name=param_name,
                                                        goal=param_value)
        self.params: Dict[str, GoalParam] = goal_params

        self.executor = executor
        self.executor_return_type: Optional[Type] = executor_return_type

        functools.update_wrapper(self, self.executor)

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
                                                        data_type=param["type"])

    def execute(self, possible_kwargs):
        run_kwargs = self.param_defaults()
        for kwarg_name, kwarg_value in possible_kwargs.items():
            if kwarg_name in self.params:
                run_kwargs[kwarg_name] = kwarg_value
        return self.executor(**run_kwargs)

    def from_json(self, json):
        pass

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

    def param_defaults(self):
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
        return self.goal_manager.run(self, **kwargs)

    def __str__(self) -> str:
        return F'{self.scope}:{self.name}' \
            if self.scope is not None else self.name


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
    pass


class DuplicateGoal(Exception):
    pass


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
        self.lib: pu.DictAccess[str, pu.DictAccess[str, GoalDefinition]] = pu.DictAccess()
        self.core_lib = {self.SCOPE_STANDARD: self.add_scope(self.SCOPE_STANDARD)}
        self.scope_default: Optional[str] = None
        
    def get_scope(self, scope: Optional[str] = None):
        return self.scope_default if scope is None else scope
    
    def add_scope(self,
                  scope: str,
                  values: Optional[pu.DictAccess[str, GoalDefinition]] = None) -> pu.DictAccess[str, GoalDefinition]:
        if scope not in self.lib:
            scoped_goals = pu.DictAccess() if values is None else values
            self.lib[scope] = scoped_goals
            return scoped_goals
        else:
            return self.lib[scope]

    def in_scope(self, scope: str):
        return GoalScope(goal_manager=self, scope=scope)

    def find(self,
             scope: Optional[str] = None,
             goals: Optional[List[Union[str, Callable, GoalDefinition]]] = None,
             fail_if_missing: bool = True) -> pu.DictAccess[str, GoalDefinition]:
        scope = self.get_scope(scope)
        
        if goals is not None:
            if not isinstance(goals, List):
                goals = [goals]
            if len(goals) == 0:
                return pu.DictAccess()

        scoped_goals = self.lib.get(scope)

        if scoped_goals is None:
            return pu.DictAccess()
        elif goals is None:
            return scoped_goals
        else:
            filtered_goals = pu.DictAccess()
            for goal in goals:
                if isinstance(goal, GoalDefinition):
                    filtered_goals[goal.name] = goal
                elif isinstance(goal, str):
                    if goal in scoped_goals:
                        filtered_goals[goal] = scoped_goals[goal]
                    elif fail_if_missing:
                        raise MissingGoal(goal)
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
            raise DuplicateGoal(goal_def.name)
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
                                      executor_return_type=pu.return_type(fun),
                                      executor_params=pu.params(fun).values(),
                                      pre=pre,
                                      post=post,
                                      params=params,
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
            name,
            scope: Optional[str] = None,
            context: Optional[GoalRunContext] = None,
            **kwargs):
        return self._run(context=context,
                         goal=name,
                         scope=scope,
                         fun_kwargs={**kwargs})

    def _run(self,
             goal: Union[str, List, Tuple, GoalDefinition] = None,
             scope: Optional[str] = None,
             context: Optional[GoalRunContext] = None,
             fun_kwargs: Optional[Dict[str, Any]] = None):

        scope = self.get_scope(scope)

        context = GoalRunContext(goal=goal,
                                 scope=scope,
                                 parent=context,
                                 phase="Run")

        self.log(context, "Start")

        if isinstance(goal, (list, tuple)):
            results = []
            for goal_name in goal:
                results.append(self._run(context=context,
                                         goal=goal_name,
                                         scope=scope,
                                         fun_kwargs=fun_kwargs))
            return results
        else:
            if isinstance(goal, GoalDefinition):
                goal_def = goal
            elif goal is not None:
                goal_def = self.get(goal, scope=scope)
            else:
                raise UnmetGoal(goal,
                                phase="run",
                                message=f"Error in run goal is missing")

            updated_kwargs = fun_kwargs.copy()

            results = []
            for pre in goal_def.pre_goals:
                try:
                    result = self._run(context=context,
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
                        updated_kwargs[param_name] = self._run(context=context,
                                                               goal=param_goal,
                                                               fun_kwargs={**param_value.goal_inputs,
                                                                           **updated_kwargs})  # todo: use strict fun kwargs?
                    else:
                        updated_kwargs[param_name] = lambda **param_invoke_kwargs: self._run(
                            context=context,
                            goal=param_goal,
                            scope=scope,
                            fun_kwargs={**param_value.goal_inputs,
                                        **updated_kwargs,
                                        **param_invoke_kwargs})
                except Exception as e:
                    raise UnmetGoal(goal,
                                    phase="param",
                                    message=f"Error in param {goal}({param_name}:{param_goal.name}) -> {e}") from e
            self.log(context, "Run")

            result = goal_def.execute(updated_kwargs)

            for post in goal_def.post_goals:
                try:
                    result = self._run(context=context,
                                       goal=post,
                                       scope=scope,
                                       fun_kwargs=fun_kwargs)
                    results.append(result)
                except Exception as e:
                    raise UnmetGoal(goal=goal, phase="post", message="Error in post") from e

            self.log(context, "End")

            return result

    def log(self, context, action):
        if self.print_logs:
            print((context.level * 5) * " ",
                  "@Goal(",
                  *((context.scope, "::") if context.scope is not None else ()),
                  context.goal, ') -> ',
                  action,
                  " Time: ", time.time() - context.start_time)

    def run_batch(self, batch=None, path=None, scope: Optional[str] = None, run_kwargs=None):
        scope = self.get_scope(scope)

        if run_kwargs is None:
            run_kwargs = {}

        if path is not None:
            batch = ju.read_file(path)
        elif isinstance(batch, str):
            batch = ju.read(batch)

        if batch is None:
            raise ValueError("json or json file required")

        if "params" in batch:
            run_kwargs.update(batch["params"])

        batch_results = []
        goals = batch["goals"]
        if isinstance(goals, dict):
            goals = [goals]
        for goal_entry in goals:
            batch_result = self._run_batch_goal(goal_entry=goal_entry,
                                                goal_args=run_kwargs,
                                                scope=scope)
            run_kwargs["__previous_result__"] = batch_result
            batch_results.append(batch_result)
        return batch_results

    def _run_batch_goal(self, goal_entry: dict, goal_args: dict, scope: Optional[str] = None):
        scope = self.get_scope(scope)

        if "params" in goal_entry:
            for param_key, param_value in goal_entry["params"].items():
                if isinstance(param_value, dict):
                    goal_args[param_key] = self.run(goal_entry["goal"], scope=scope, **goal_args)
                else:
                    goal_args[param_key] = param_value
            goal_args.update()
        if "argParams" in goal_entry:
            for arg_param_key, arg_param_value in goal_entry["argParams"].items():
                goal_args[arg_param_key] = goal_args[arg_param_value]

        if "loop" in goal_entry:
            loop_entry = goal_entry["loop"]
            if "goals" in loop_entry:
                goals = loop_entry["goals"]
            elif "goal" in loop_entry:
                goals = [loop_entry["goal"]]
            else:
                raise ValueError("Loop: missing goal(s)", goal_entry)
            arg_name = loop_entry["arg"] if "arg" in loop_entry else "__loop_arg__"

            if "range" in loop_entry:
                loop_over = range(loop_entry["range"])
            elif "values" in loop_entry:
                loop_over = loop_entry["values"]
            elif "valuesGoal" in loop_entry:
                loop_over = self.run(loop_entry["valuesGoal"], scope=scope, **goal_args)
            elif "valuesArg" in loop_entry:
                loop_over = goal_args[loop_entry["valuesArg"]]
            else:
                raise ValueError("Loop: missing loop over", goal_entry)

            loop_result = self._run_batch_goal_loop(loop_over=loop_over,
                                                    goals=goals,
                                                    goal_args=goal_args,
                                                    loop_arg_name=arg_name,
                                                    scope=scope)
            goal_args["__previous_result__"] = loop_result

            if "resultArg" in loop_entry:
                goal_args[loop_entry["resultArg"]] = loop_result

            if "resultGoal" in loop_entry:
                if "resultGoalParam" not in loop_entry:
                    raise ValueError("resultGoal requires resultGoalParam", loop_entry)

                return self.run(loop_entry["resultGoal"],
                                scope=scope,
                                **{**goal_args, loop_entry["resultGoalParam"]: loop_result})
            else:
                return loop_result
        elif "goal" in goal_entry:
            return self.run(goal_entry["goal"], scope=scope, **goal_args)
        else:
            raise ValueError("Invalid batch Goal", goal_entry)

    def _run_batch_goal_loop(self, loop_over, goals: list, goal_args: dict, loop_arg_name: str, scope: Optional[str] = None):
        scope = self.get_scope(scope)

        if loop_over is None:
            return
        if not pu.iterable(loop_over):
            loop_over = [loop_over]

        loop_results = []
        for arg in loop_over:
            for loop_goal_entry in goals:
                loop_result = self._run_batch_goal(goal_entry=loop_goal_entry,
                                                   goal_args={**goal_args, loop_arg_name: arg},
                                                   scope=scope)
                loop_results.append(loop_result)

        return loop_results

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
        parser.add_argument("--goal_json",
                            type=str,
                            help="Goal JSON file, command line params will override")
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
        args = pu.parse_unknown_args(arg_parser)
        args = pu.remove_na_from_dict(args)
        # for goal_name in goal_names:
        #     self._create_arg_parser_params(arg_parser=arg_parser,
        #                                    default_goal=goal_name,
        #                                    scope=scope,
        #                                    custom_inputs=custom_inputs,
        #                                    enforced_required=True)
        # args = arg_parser.parse_args()

        if "goal_json" in args:
            goal_json = args["goal_json"]
            del args['goal_json']
            result = self.run_batch(path=goal_json, scope=scope, run_kwargs={**kwargs, **args})
        elif "goal" in args:
            goal_name = args["goal"]
            goal_names = goal_name.split(" ")
            del args['goal']

            result = self.run(goal_names, scope=scope, **{**kwargs, **args})
        else:
            raise ValueError("no goal or json specified")

        if print_result and result is not None:
            print(result)

        return result

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
