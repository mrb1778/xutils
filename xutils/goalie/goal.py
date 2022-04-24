import argparse
import functools
from types import FunctionType
from typing import Union, Any, Dict, List

import xutils.core.python_utils as pu
import xutils.data.json_utils as ju


class GoalDefinition:
    def __init__(self, name, scope=None, pre=None, params=None, refs=None, fun=None, goal_lookup=None) -> None:
        super().__init__()

        self.scope = scope
        self.name = name

        if pre is None:
            pre = []
        if not isinstance(pre, (list, tuple)):
            pre = [pre]
        for i, ref_goal in enumerate(pre):
            if isinstance(ref_goal, FunctionType):
                pre[i] = goal_lookup(ref_goal)
        self.pre = pre

        if params is None:
            params = dict()
        for param_name, param_value in params.items():
            if isinstance(param_value, (str, FunctionType)):
                if isinstance(param_value, FunctionType):
                    param_value = goal_lookup(param_value)
                params[param_name] = GoalParam(param_value)
            elif isinstance(param_value, GoalParam) and param_value.fun_def is not None:
                param_value.name = goal_lookup(param_value.fun_def)
        self.params = params

        if refs is None:
            refs = []
        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        for i, ref_goal in enumerate(refs):
            if isinstance(ref_goal, FunctionType):
                refs[i] = goal_lookup(ref_goal)
        self.refs = refs

        if fun is not None:
            for param in pu.params(fun).values():
                param_name = param["name"]
                if param_name not in params:
                    if param_name == StackParam.DEFAULT_NAME:
                        params[param_name] = StackParam(param_name)
                    else:
                        params[param_name] = InputParam(name=param_name,
                                                        required=param["required"],
                                                        default=param["default"],
                                                        data_type=param["type"])
        self.fun = fun

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
    def __init__(self, name: str, default=None, label=None, description=None, data_type=None, required=False) -> None:
        super().__init__(name)
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
        return json


class GoalParam(ParamType):
    def __init__(self, name: Any, **kwargs) -> None:
        super().__init__(name if isinstance(name, str) else name.__name__)
        if isinstance(name, FunctionType):
            self.fun_def = name
        else:
            self.fun_def = None
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
    pass


class GoalRegistry:
    SCOPE_ALL = "__all__"
    SCOPE_DEFAULT = "__default__"

    def __init__(self):
        self._goals: Dict[str, Dict[str, GoalDefinition]] = {}
        self.current_scope = GoalScope(self, scope=self.SCOPE_DEFAULT)

    def goals(self, scope=None, create=False):
        if scope is None:
            scope = self.current_scope.scope

        if scope not in self._goals:
            if create:
                scoped_goals = {}
                self._goals[scope] = scoped_goals
                return scoped_goals
        else:
            return self._goals[scope]

    def goal_names(self, scope=None):
        return list(self.goals(scope).keys())

    def goal_definitions(self, scope=None):
        return list(self.goals(scope).values())

    def _get_goal_for_fun(self, fun: FunctionType, scope=None):
        goal_name = fun.__name__
        assert self.definition(goal_name, scope=scope)
        return goal_name

    def reset(self, scope=None):
        if scope == self.SCOPE_ALL or (scope is None and self.current_scope.scope == self.SCOPE_DEFAULT):
            self._goals.clear()
        elif scope in self._goals:
            self._goals[scope].clear()

    def scope(self, scope):
        return GoalScope(self, scope=scope)

    def add(self,
            name=None,
            fun=None,
            pre=None,
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
                                      params=params,
                                      fun=fun,
                                      refs=refs,
                                      goal_lookup=lambda find_name: self._get_goal_for_fun(find_name, scope=scope))

        scoped_goals = self.goals(scope=scope, create=True)
        if not overwrite and goal_def.name in scoped_goals:
            raise DuplicateGoal(goal_def.name)

        scoped_goals[goal_def.name] = goal_def

    def __call__(self, name: str, pre=None, params=None, refs=None, scope=None):
        def decorator_goal(fun):
            self.add(name, fun, pre, params, refs, scope=scope)

            @functools.wraps(fun)
            def wrapper(**kwargs):
                return self.run(name, scope=scope, **kwargs)

            return wrapper

        return decorator_goal

    def definition(self, name, scope=None, fail_if_missing=True) -> GoalDefinition:
        scoped_goals = self.goals(scope=scope)
        if fail_if_missing and (scoped_goals is None or name not in scoped_goals.keys()):
            raise MissingGoal(name)

        return scoped_goals[name]

    def requirements(self, name, scope=None):
        goal_def = self.definition(name, scope=scope)

        full_requirements = {name}

        for pre in goal_def.pre:
            full_requirements.update(self.requirements(pre, scope=scope))

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
        if isinstance(name, (list, tuple)):
            results = []
            for goal_name in name:
                results.append(self.run(goal_name, scope=scope, **kwargs))
            return results
        else:
            goal_def = self.definition(name, scope=scope)

            updated_kwargs = kwargs.copy()

            results = []
            for prerequisite in goal_def.pre:
                try:
                    result = self.run(prerequisite, scope=scope, **kwargs)
                    results.append(result)
                except Exception as e:
                    raise UnmetGoal(name) from e

            if len(results) > 0:
                updated_kwargs[ResultParam.ALL_KWARGS_PARAM] = results[0]
            updated_kwargs[ResultsParam.ALL_KWARGS_PARAM] = results

            for param_name, param_value in goal_def.params_of_type(GoalParam).items():
                goal_name = param_value.name
                try:
                    fun_kwargs = {**param_value.kwargs, **kwargs}
                    updated_kwargs[param_name] = self.run(goal_name, scope=scope, **fun_kwargs)
                except Exception as e:
                    raise UnmetGoal(name) from e

            return goal_def.execute(updated_kwargs)

    def run_batch(self, json=None, json_path=None, scope=None, base_args=None, **kwargs):
        if json is not None:
            json = ju.read(json)
        elif json_path is not None:
            json = ju.read_file(json_path)

        if json is None:
            raise ValueError("json or json file required")

        root_args = {}
        if base_args is not None:
            root_args.update(base_args)

        if "args" in json:
            root_args.update(json["args"])

        for goal_entry in json["goals"]:
            self._run_batch_goal(goal_entry=goal_entry,
                                 goal_args={**root_args, **kwargs},
                                 scope=scope)

    def _run_batch_goal(self, goal_entry, goal_args, scope):

        if "args" in goal_entry:
            goal_args.update(goal_entry["args"])

        if "loop" in goal_entry:
            loop_entry = goal_entry["loop"]
            arg_name = loop_entry["arg"]
            goals = loop_entry["goals"]

            if "goal" in loop_entry:
                loop_over = self.run(loop_entry["goal"], scope=scope, **goal_args)
                self._run_batch_goal_loop(loop_over=loop_over,
                                          goals=goals,
                                          goal_args=goal_args,
                                          loop_arg_name=arg_name,
                                          scope=scope)
            elif "values" in loop_entry:
                self._run_batch_goal_loop(loop_over=loop_entry["values"],
                                          goals=goals,
                                          goal_args=goal_args,
                                          loop_arg_name=arg_name,
                                          scope=scope)

            else:
                raise ValueError("Invalid Loop Goal/Args", goal_entry)

        elif "goal" in goal_entry:
            self.run(goal_entry["goal"], scope=scope, **goal_args)
        else:
            raise ValueError("Invalid batch Goal", goal_entry)

    def _run_batch_goal_loop(self, loop_over, goals, goal_args, loop_arg_name, scope):
        if loop_over is None:
            return
        if not pu.iterable(loop_over):
            loop_over = [loop_over]
        for arg in loop_over:
            print("_run_batch_goal_loop", arg)
            for loop_goal_entry in goals:
                self._run_batch_goal(goal_entry=loop_goal_entry,
                                     goal_args={**goal_args, loop_arg_name: arg},
                                     scope=scope)

    def to_json(self, scope=None):
        return {
            "goals": [g.to_json() for g in self.goal_definitions(scope=scope)]
        }

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
        goal_names = None
        if "goal" in args:
            goal_name = args["goal"]
            goal_names = goal_name.split(" ")
            del args['goal']

        goal_json = None
        if "goal_json" in args:
            goal_json = args["goal_json"]
            del args['goal_json']

        # for goal_name in goal_names:
        #     self._create_arg_parser_params(arg_parser=arg_parser,
        #                                    default_goal=goal_name,
        #                                    scope=scope,
        #                                    custom_inputs=custom_inputs,
        #                                    enforced_required=True)
        # args = arg_parser.parse_args()

        if goal_json is not None:
            result = goal.run_batch(json_path=goal_json, scope=scope, base_args=kwargs, **args)
        elif goal_names is not None:
            result = goal.run(goal_names, scope=scope, **{**kwargs, **args})
        else:
            raise ValueError("no goal or json specified")

        if print_result and result is not None:
            print(result)

        return result


class GoalScope:
    def __init__(self, goal_registry: GoalRegistry, scope: str):
        self.goal_registry = goal_registry
        self.scope = scope
        self.previous_scope = None

    def __enter__(self):
        self.activate()

    def activate(self):
        self.previous_scope = self.goal_registry.current_scope
        self.goal_registry.current_scope = self

    def __exit__(self, *args):
        self.deactivate()

    def deactivate(self):
        self.goal_registry.current_scope = self.previous_scope


goal = GoalRegistry()