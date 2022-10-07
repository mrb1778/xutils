import argparse
import functools
import time
from types import FunctionType
from typing import Union, Any, Dict, List, Callable, Literal

import xutils.core.python_utils as pu
import xutils.data.json_utils as ju


class GoalIdentifier:
    def __init__(self, name: str, scope: str) -> None:
        super().__init__()
        self.name = name
        self.scope = scope

    def __str__(self) -> str:
        return F'{self.scope}:{self.name}' \
            if self.scope is not None else self.name


def _get_goal_id(goal_id, scope):
    if isinstance(goal_id, GoalIdentifier):
        return goal_id
    elif isinstance(goal_id, FunctionType):
        if goal_id.__goal_identifier__ is None:
            raise MissingGoal(goal_id)
        return goal_id.__goal_identifier__
    elif isinstance(goal_id, str):
        return GoalIdentifier(goal_id, scope)


def _get_goal_ids(goals, scope):
    if goals is None:
        return []
    else:
        if not isinstance(goals, (list, tuple)):
            goals = [goals]
        return [_get_goal_id(g, scope) for g in goals]


class GoalDefinition:
    def __init__(self, name, scope=None, pre=None, post=None, params=None, refs=None, fun=None) -> None:
        super().__init__()

        self.scope = scope
        self.name = name

        self.pre = _get_goal_ids(pre, scope)
        self.post = _get_goal_ids(post, scope)

        if params is None:
            self.params = dict()
        else:
            self.params = {
                param_name: param_value
                if isinstance(param_value, ParamType)
                else GoalParam(_get_goal_id(param_value, scope))
                for param_name, param_value in params.items()
            }
            # for param_name, param_value in params.items():
            #     if isinstance(param_value, (str, FunctionType)):
            #         if isinstance(param_value, FunctionType):
            #             param_value = goal_lookup(param_value)
            #         params[param_name] = GoalParam(param_value)
            #     elif isinstance(param_value, GoalParam) and param_value.fun_def is not None:
            #         param_value.name = goal_lookup(param_value.fun_def)
            # self.params = params

        self.refs = _get_goal_ids(refs, scope)

        if fun is not None:
            for param in pu.params(fun).values():
                param_name = param["name"]
                if param_name not in self.params:
                    if param_name == StackParam.DEFAULT_NAME:
                        self.params[param_name] = StackParam(param_name)
                    else:
                        self.params[param_name] = InputParam(name=param_name,
                                                             required=param["required"],
                                                             default=param["default"],
                                                             data_type=param["type"])

        self.fun = fun

        input_params = self.params_of_type(InputParam)
        for input_param in input_params.values():
            input_param.source = self.name

    def execute(self, possible_kwargs):
        reg_fun = self.fun
        fun_params = pu.param_defaults(reg_fun)
        filtered_fun_kwargs = {param_name: param_value
                               for param_name, param_value in possible_kwargs.items()
                               if param_name in fun_params}

        for param_name in self.params_of_type(StackParam):
            filtered_fun_kwargs[param_name] = GoalStack(scope=self.scope, **possible_kwargs)

        for param_name in self.params_of_type(AllParams):
            filtered_fun_kwargs[param_name] = possible_kwargs

        for param_name in self.params_of_type(ResultParam) and ResultParam.ALL_KWARGS_PARAM in possible_kwargs:
            filtered_fun_kwargs[param_name] = possible_kwargs[ResultParam.ALL_KWARGS_PARAM]

        for param_name in self.params_of_type(ResultsParam) and ResultsParam.ALL_KWARGS_PARAM in possible_kwargs:
            filtered_fun_kwargs[param_name] = possible_kwargs[ResultsParam.ALL_KWARGS_PARAM]

        return reg_fun(**filtered_fun_kwargs)

    def from_json(self, json):
        pass

    def params_of_type(self, param_type):
        return {param_name: param_value
                for param_name, param_value in self.params.items()
                if isinstance(param_value, param_type)}

    def to_json(self):
        json = {
            "name": self.name
        }
        if len(self.pre):
            json["pre"] = self.pre

        if len(self.post):
            json["post"] = self.post

        if len(self.params):
            json["params"] = [
                p.to_json() for p in self.params.values()
            ]
        return json


class GoalStack(object):
    def __init__(self, scope=None, **kwargs) -> None:
        super().__init__()
        self.scope = scope
        self.kwargs = kwargs

    def __getattr__(self, goal_name, **kwargs):
        return goal.run(goal_name, scope=self.scope, **{**self.kwargs, **kwargs})


class ParamType:
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def to_json(self):
        return {
            "name": self.name,
            "type": self.__class__.__name__
        }
    # def __repr__(self):
    #     return f"{self.__class__.__name__}({self.name})"
    #
    # def __str__(self):
    #     return self.name


class InputParam(ParamType):
    def __init__(self,
                 name: str,
                 default=None,
                 label=None,
                 description=None,
                 data_type=None,
                 required=False,
                 source=None) -> None:
        super().__init__(name)
        self.source = source
        self.default = default
        self.label = label
        self.description = description
        self.required = required

        if data_type is not None:
            self.data_type = data_type
        elif self.default is not None:
            self.data_type = type(self.default)
        else:
            self.data_type = str

    def to_json(self):
        json = super().to_json()
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
        if self.source is not None:
            json["source"] = self.source
        return json


class GoalParam(ParamType):
    def __init__(self, goal_id: Any, scope=None, **kwargs) -> None:
        super().__init__(goal_id)
        self.goal_identifier = _get_goal_id(goal_id, scope)

        self.kwargs = kwargs

    def to_json(self):
        json = super().to_json()
        if self.kwargs is not None and len(self.kwargs):
            json["kwargs"] = self.kwargs

        return json


class StackParam(ParamType):
    DEFAULT_NAME = "stack"

    def __init__(self, name: str) -> None:
        super().__init__(name)


class AllParams(ParamType):
    DEFAULT_NAME = "params"

    def __init__(self, name: str) -> None:
        super().__init__(name)


class ResultParam(ParamType):
    DEFAULT_NAME = "result"
    ALL_KWARGS_PARAM = "__RESULT_PARAM__"

    def __init__(self, name: str) -> None:
        super().__init__(name)


class ResultsParam(ParamType):
    DEFAULT_NAME = "results"
    ALL_KWARGS_PARAM = "__RESULTS_PARAM__"

    def __init__(self, name: str) -> None:
        super().__init__(name)


class MissingGoal(Exception):
    pass


class DuplicateGoal(Exception):
    pass


class UnmetGoal(Exception):
    def __init__(self, name, phase: Literal["pre", "param", "post"], message: str) -> None:
        super().__init__(name, phase, message)
        self.name = name
        self.phase = phase
        self.message = message

    def __str__(self) -> str:
        str_value = f"UnmetGoal({self.phase}:{self.name})\n-->{self.__cause__.__str__()}\n"
        if isinstance(self.__cause__, self.__class__):
            str_value += self.message
        return str_value


class GoalRegistry:
    SCOPE_ALL = "__all__"
    SCOPE_DEFAULT = "__default__"

    def __init__(self):
        self.trace = False
        self._goals: Dict[str, Dict[str, GoalDefinition]] = {}
        # self.current_scope = GoalScope(self, scope=self.SCOPE_DEFAULT)

    def goals(self, scope=None, create=False):
        if scope is None:
            # scope = self.current_scope.scope
            scope = self.SCOPE_DEFAULT

        if scope not in self._goals:
            scoped_goals = {}
            if create:
                self._goals[scope] = scoped_goals
            return scoped_goals
        else:
            return self._goals[scope]

    def goal_names(self, scope=None):
        return list(self.goals(scope).keys())

    def goal_definitions(self, scope=None):
        return list(self.goals(scope).values())

    def _get_goal_for_fun(self, fun: FunctionType):
        goal_id = fun.__goal_identifier__
        if self.definition(goal_id) is None:
            raise MissingGoal(goal_id)
        return goal_id

    def reset(self, scope=None):
        if scope == self.SCOPE_ALL or scope is None:
            #  and self.current_scope.scope == self.SCOPE_DEFAULT):
            self._goals.clear()
        elif scope in self._goals:
            self._goals[scope].clear()

    # def scope(self, scope):
    #     return GoalScope(self, scope=scope)

    def add(self,
            name=None,
            fun=None,
            pre=None,
            post=None,
            params=None,
            refs=None,
            overwrite=False,
            scope=None,
            goal_def: Union[dict, GoalDefinition] = None):
        if goal_def is not None:
            if isinstance(goal_def, dict):
                goal_def = GoalDefinition(**goal_def)
            elif not isinstance(goal_def, GoalDefinition):
                raise MissingGoal("Invalid Goal Definition", goal_def)
        else:
            goal_def = GoalDefinition(name=name,
                                      scope=scope,
                                      pre=pre,
                                      post=post,
                                      params=params,
                                      fun=fun,
                                      refs=refs)

        scoped_goals = self.goals(scope=scope, create=True)
        if not overwrite and goal_def.name in scoped_goals:
            raise DuplicateGoal(goal_def.name)

        scoped_goals[goal_def.name] = goal_def

    def __call__(self, function: Callable = None, name: str = None, pre=None, post=None, params=None, refs=None, scope=None):
        def decorator_goal(fun: Callable):
            nonlocal name
            if name is None:
                name = fun.__name__

            goal_identifier = GoalIdentifier(name, scope)
            fun.__goal_identifier__ = goal_identifier
            self.add(name,
                     fun=fun,
                     pre=pre,
                     post=post,
                     params=params,
                     refs=refs,
                     scope=scope)

            @functools.wraps(fun)
            def wrapper(**kwargs):
                return self.run(name, scope=scope, **kwargs)

            wrapper.__goal_identifier__ = goal_identifier
            return wrapper

        if function:
            return decorator_goal(function)
        return decorator_goal

    def definition(self, name, scope=None, fail_if_missing=True) -> GoalDefinition:
        if isinstance(name, FunctionType):
            name = self._get_goal_for_fun(name)
        if isinstance(name, GoalIdentifier):
            scope = name.scope
            name = name.name
        scoped_goals = self.goals(scope=scope)
        if fail_if_missing and (scoped_goals is None or name not in scoped_goals.keys()):
            raise MissingGoal(name)

        return scoped_goals[name]

    def requirements(self, name, scope=None):
        goal_def = self.definition(name, scope=scope)

        full_requirements = {name.name if isinstance(name, GoalIdentifier) else name}

        for pre in goal_def.pre:
            full_requirements.update(self.requirements(pre, scope=scope))

        for post in goal_def.post:
            full_requirements.update(self.requirements(post, scope=scope))

        for ref in goal_def.refs:
            full_requirements.update(self.requirements(ref, scope=scope))

        for param_name, param_value in goal_def.params_of_type(GoalParam).items():
            goal_name = param_value.name
            full_requirements.update(self.requirements(goal_name, scope=scope))

        return full_requirements

    def leafs(self, scope=None):
        #  todo: optimize and return 1st one instead
        all_goals = self.goals(scope=scope)
        leafs = []
        for goal_name in all_goals.keys():
            if len(self.requirements(goal_name, scope=scope)) == 0:
                leafs.append(goal_name)

        return leafs

    def roots(self, scope=None):
        #  todo: optimize
        all_goals = self.goal_names(scope=scope)
        all_requirements = {}

        for goal_name in all_goals:
            all_goals.extend(self.requirements(goal_name, scope=scope))

        return [goal_name for goal_name in all_goals if goal_name not in all_requirements]

    def inputs(self, name=None, scope=None):
        return self.params_of_type(param_type=InputParam, name=name, scope=scope)

    def params_of_type(self, param_type, name=None, scope=None):
        requirements = set()
        if name is None:
            for goal_name in self.goal_names(scope=scope):
                requirements.update(self.requirements(goal_name, scope=scope))
        else:
            requirements = self.requirements(name, scope=scope)

        found_params = {}
        for requirement in requirements:
            goal_def = self.definition(requirement, scope=scope)
            found_params.update(goal_def.params_of_type(param_type))

        return list(found_params.values())

    def run(self, name, scope=None, **kwargs):
        return self._run_with_context(name=name,
                                      scope=scope,
                                      fun_kwargs={**kwargs})

    def _run_with_context(self, name, scope=None, goal_context=None, fun_kwargs=None):
        if goal_context is None:
            goal_context = {"level": 0, "Phase": "Run"}
        else:
            goal_context = {**goal_context, "level": goal_context["level"] + 1}

        if self.trace:
            print((goal_context["level"] * 5) * " ",
                  "@Goal(",
                  *((scope, "::") if scope is not None else ()),
                  name, ')')
            start_time = time.time()

        if isinstance(name, (list, tuple)):
            results = []
            for goal_name in name:
                results.append(self._run_with_context(goal_context=goal_context,
                                                      name=goal_name,
                                                      scope=scope,
                                                      fun_kwargs=fun_kwargs))
            return results
        else:
            goal_def = self.definition(name, scope=scope)

            updated_kwargs = fun_kwargs.copy()

            results = []
            for pre in goal_def.pre:
                try:
                    result = self._run_with_context(goal_context=goal_context,
                                                    name=pre,
                                                    scope=scope,
                                                    fun_kwargs=fun_kwargs)
                    results.append(result)
                except Exception as e:
                    raise UnmetGoal(name=name, phase="pre", message="Pre Failed") from e

            if len(results) > 0:
                updated_kwargs[ResultParam.ALL_KWARGS_PARAM] = results[0]
            updated_kwargs[ResultsParam.ALL_KWARGS_PARAM] = results

            for param_name, param_value in goal_def.params_of_type(GoalParam).items():
                goal_identifier = param_value.goal_identifier
                try:
                    fun_kwargs = {**param_value.kwargs, **fun_kwargs}
                    updated_kwargs[param_name] = self._run_with_context(goal_context=goal_context,
                                                                        name=goal_identifier,
                                                                        scope=scope,
                                                                        fun_kwargs=fun_kwargs)
                except Exception as e:
                    raise UnmetGoal(name,
                                    phase="param",
                                    message=f"Error in param {name}({param_name}:{goal_identifier}) -> {e}") from e

            if self.trace:
                print((goal_context["level"] * 5) * " ", "@Goal Run", name)

            result = goal_def.execute(updated_kwargs)

            for post in goal_def.post:
                try:
                    result = self._run_with_context(goal_context=goal_context,
                                                    name=post,
                                                    scope=scope,
                                                    fun_kwargs=fun_kwargs)
                    results.append(result)
                except Exception as e:
                    raise UnmetGoal(name=name, phase="post", message="Error in post") from e

            if self.trace:
                print((goal_context["level"] * 5) * " ", "@Goal Duration", time.time() - start_time)

            return result

    def run_batch(self, batch=None, path=None, scope=None, run_kwargs=None):
        if run_kwargs is None:
            batch_kwargs = {}

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

    def _run_batch_goal(self, goal_entry: dict, goal_args: dict, scope: str):
        if "params" in goal_entry:
            goal_args.update(goal_entry["params"])
        if "argParams" in goal_entry:
            print("argParams:goal_args", goal_args.keys())
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

    def _run_batch_goal_loop(self, loop_over, goals: list, goal_args: dict, loop_arg_name: str, scope:str):
        if loop_over is None:
            return
        if not pu.iterable(loop_over):
            loop_over = [loop_over]

        loop_results = []
        for arg in loop_over:
            print("_run_batch_goal_loop", arg)
            for loop_goal_entry in goals:
                loop_result = self._run_batch_goal(goal_entry=loop_goal_entry,
                                                   goal_args={**goal_args, loop_arg_name: arg},
                                                   scope=scope)
                loop_results.append(loop_result)

        return loop_results

    def to_json(self, scope=None):
        return {
            "goals": [g for g in self.goal_definitions(scope=scope)]
        }

    def export_goals(self, file_name: str):
        ju.write_to(self, path=file_name)

    def create_arg_parser(self,
                          default_goal=None,
                          all_args=True,
                          scope=None,
                          custom_inputs=None,
                          **kwargs):
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
                                  scope=None,
                                  custom_inputs=None,
                                  enforced_required=False,
                                  all_args=True,
                                  **kwargs):
        # all_goals = self.goal_names(scope=scope)
        # root_goals = self.roots(scope=scope)
        # for root_goal in root_goals:
        #     del all_goals[root_goal]
        #     all_goals.append(f"{root_goal}*")

        goal_args_group = arg_parser.add_argument_group("All Args" if all_args else f"{default_goal} Args")
        inputs = self.inputs(name=None if all_args else default_goal, scope=scope)
        for goal_input in inputs:
            input_name = goal_input.name
            if custom_inputs is not None and input_name in custom_inputs:
                goal_input = custom_inputs[input_name]

            input_default = kwargs[input_name] if input_name in kwargs else goal_input.default

            help_text = ""
            if goal_input.description is not None:
                help_text = goal_input.description
            if input_default is not None:
                help_text += " (default: %(default)s)"

            goal_args_group.add_argument(f"--{input_name}",
                                         default=input_default,
                                         type=goal_input.data_type if goal_input.data_type != List else None,
                                         # required=True if input_default is None else False,
                                         required=goal_input.required if enforced_required else False,
                                         nargs="*" if goal_input.data_type == List else None,
                                         action="store",
                                         help=help_text)

    def run_from_arg_parse(self,
                           arg_parser: argparse.ArgumentParser = None,
                           default_goal=None,
                           all_args=True,
                           scope=None,
                           custom_inputs=None,
                           print_result=True,
                           **kwargs):
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
            result = goal.run_batch(path=goal_json, scope=scope, run_kwargs={**kwargs, **args})
        elif "goal" in args:
            goal_name = args["goal"]
            goal_names = goal_name.split(" ")
            del args['goal']

            result = goal.run(goal_names, scope=scope, **{**kwargs, **args})
        else:
            raise ValueError("no goal or json specified")

        if print_result and result is not None:
            print(result)

        return result

    def debug(self, on: bool = True) -> None:
        self.trace = on

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


goal = GoalRegistry()
