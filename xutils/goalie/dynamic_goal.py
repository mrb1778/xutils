from typing import Union, Optional, Dict, Any, List
import re

import xutils.data.config_utils as cu
import xutils.data.json_utils as ju

from .goal import GoalDefinition, GoalManager, GoalRef
from ..expressions.expression_evaluator import DictVariableResolver
from ..expressions.sexpression import SimpleExpressionEvaluator


class DynamicGoal(GoalDefinition):
    PARAMS_KEY = "params"
    PRE_KEY = "pre"

    def __init__(self,
                 name: str,
                 scope: Optional[str] = None,
                 goal_manager: GoalManager = None,
                 exec_goal: Union[str, GoalDefinition] = None,
                 goal_def: Union[str, Dict[str, Any]] = None,
                 run_regex: str = r'=(.*)',
                 ref_regex: str = r'\&(.*)',
                 meta_param: str = '@') -> None:
        super().__init__(name=name, scope=scope, goal_manager=goal_manager)
        self.run_regex: re.Pattern = re.compile(run_regex)
        self.ref_regex: re.Pattern = re.compile(ref_regex)
        self.meta_param: str = meta_param
        self.exec_goal = exec_goal
        self.goal_def = goal_def
        self._parse_meta()

        self.expression_evaluator = SimpleExpressionEvaluator()

        # self._populate_params()

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
                if self.PRE_KEY in self.meta_params:
                    self.pre_goals = list(self.goal_manager.find(scope=self.scope, goals=self.meta_params["pre"]).values())

    # def _populate_params(self):
    #     self._parse_params(self.goal_def)
    #
    # def _parse_params(self, what):
    #     if isinstance(what, GoalDefinition):
    #         self.params.update(what.params)
    #     elif isinstance(what, str):
    #         # run_regex_check = re.fullmatch(self.run_regex, what)
    #         # ref_regex_check = re.fullmatch(self.ref_regex, what)
    #         run_param_regex_check = re.fullmatch(self.run_param_regex, what)
    #         param_regex_check = re.fullmatch(self.param_regex, what)
    #
    #         # if ref_regex_check is not None:
    #         #     self.params.update(self.goal_manager.get(ref_regex_check.group(1)).params)
    #         # elif run_regex_check is not None:
    #         #     self.params.update(self.goal_manager.get(run_regex_check.group(1)).params)
    #         # elif param_regex_check is not None:
    #         if run_param_regex_check is not None:
    #             self.params[run_param_regex_check.group(1)] = GoalParam(name=run_param_regex_check.group(1),
    #                                                                     required=False)
    #         if param_regex_check is not None:
    #             self.params[param_regex_check.group(1)] = GoalParam(name=param_regex_check.group(1),
    #                                                                 required=False)
    #     elif isinstance(what, dict):
    #         first_key = next(iter(what))
    #         run_regex_check = re.fullmatch(self.run_regex, first_key)
    #         ref_regex_check = re.fullmatch(self.ref_regex, first_key)
    #         if len(what) == 1 and (run_regex_check is not None or ref_regex_check is not None):
    #             for goal_name, goal_params in what.items():
    #                 for param_name, param_value in goal_params.items():
    #                     self._populate_params(param_name)
    #                     self._populate_params(param_value)
    #                 if run_regex_check is not None:
    #                     self.params.update(self.goal_manager.get(run_regex_check.group(1)).params)
    #                 else:
    #                     self.params.update(self.goal_manager.get(ref_regex_check.group(1)).params)
    #         else:
    #             for key, value in what.items():
    #                 self._populate_params(key)
    #                 self._populate_params(value)
    #     elif isinstance(what, List):
    #         for item in what:
    #             self._populate_params(item)

    def execute(self, possible_kwargs: Dict[str, Any]):
        dynamic_result = self._dynamic_execute(self.goal_def, possible_kwargs)
        if self.exec_goal is not None:
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
        what = self.expression_evaluator.eval(what, DictVariableResolver(possible_kwargs))

        if not isinstance(what, str):
            return what
        else:
            run_regex_check = re.fullmatch(self.run_regex, what)
            ref_regex_check = re.fullmatch(self.ref_regex, what)
            if ref_regex_check is not None:
                return self.goal_manager.run_later(
                    goal=DynamicGoal(name=f"{self.name}:RunLater:{ref_regex_check.group(1)}",
                                     goal_def=f"={ref_regex_check.group(1)}",
                                     goal_manager=self.goal_manager),
                    fun_kwargs=possible_kwargs.copy())
            elif run_regex_check is not None:
                run_goal = run_regex_check.group(1)
                return self.goal_manager.run(run_goal, fun_kwargs=possible_kwargs)
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
            return dynamic_goal.run_later(**possible_kwargs)

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


def run_batch(goal_manager: GoalManager,
              batch=None,
              path=None,
              scope: Optional[str] = None,
              run_kwargs=None):
    scope = goal_manager.get_scope(scope)

    if run_kwargs is None:
        run_kwargs = {}

    if path is not None:
        batch = cu.read_file(path)
    elif isinstance(batch, str):
        batch = ju.read(batch)

    if batch is None:
        raise ValueError("json string or file required")

    if "debug" in batch and not goal_manager.print_logs:
        goal_manager.debug(batch["debug"])

    if "params" in batch:
        run_kwargs.update(batch["params"])

    if "goals" in batch:
        goals = batch["goals"]
        for goal_name, goal_def in goals.items():
            goal_manager.add(DynamicGoal(name=goal_name,
                                         goal_def=goal_def,
                                         goal_manager=goal_manager))

    if "run" in batch:
        run = batch["run"]
        return goal_manager.run(DynamicGoal(name=":BatchGoal:",
                                            goal_def=run,
                                            goal_manager=goal_manager),
                                fun_kwargs=run_kwargs,
                                scope=scope)
