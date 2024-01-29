import argparse
from typing import Optional, Dict, List

import xutils.core.python_utils as pyu

from .dynamic_goal import run_batch
from .goal import GoalManager, GoalParam


def run_from_arg_parse(goal_manager: GoalManager,
                       arg_parser: argparse.ArgumentParser = None,
                       default_goal=None,
                       all_args: bool = True,
                       scope: Optional[str] = None,
                       custom_inputs: Optional[Dict[str, GoalParam]] = None,
                       print_result: bool = True,
                       **kwargs):
    scope = goal_manager.get_scope(scope)

    if arg_parser is None:
        arg_parser = create_arg_parser(goal_manager=goal_manager,
                                       default_goal=default_goal,
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
            goal_manager.debug(args["debug"])
    if "goal_config" in args:
        goal_config = args["goal_config"]
        del args['goal_config']
        result = run_batch(goal_manager=goal_manager,
                           path=goal_config,
                           scope=scope,
                           run_kwargs={**kwargs, **args})
    elif "goal" in args:
        goal_name = args["goal"]
        # goal_names = goal_name.split(" ")
        del args['goal']

        if args["inspect"]:
            print(*(next(iter(goal_manager.find(goals=[goal_name], scope=scope).values())).referenced()))
        else:
            result = goal_manager.run(goal_name, scope=scope, **{**kwargs, **args})
    else:
        raise ValueError("no goal or json specified")

    if print_result and result is not None:
        print(result)

    return result


def create_arg_parser(goal_manager: GoalManager,
                      default_goal=None,
                      all_args=True,
                      scope: Optional[str] = None,
                      custom_inputs: Dict[str, GoalParam] = None,
                      **kwargs):
    scope = goal_manager.get_scope(scope)

    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", "--g",
                        type=str,
                        default=default_goal,
                        help="Which goal to execute (default: %(default)s). Options: " +
                             ', '.join(goal_manager.goal_names(scope=scope)))
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
    _create_arg_parser_params(goal_manager=goal_manager,
                              arg_parser=parser,
                              default_goal=default_goal,
                              all_args=all_args,
                              scope=scope,
                              custom_inputs=custom_inputs,
                              **kwargs)
    return parser


def _create_arg_parser_params(goal_manager: GoalManager,
                              arg_parser: argparse.ArgumentParser,
                              default_goal,
                              scope: Optional[str] = None,
                              custom_inputs: Dict[str, GoalParam] = None,
                              enforced_required=False,
                              all_args=True,
                              **kwargs):
    scope = goal_manager.get_scope(scope)

    # all_goals = self.goal_names(scope=scope)
    # root_goals = self.roots(scope=scope)
    # for root_goal in root_goals:
    #     del all_goals[root_goal]
    #     all_goals.append(f"{root_goal}*")

    goal_args_group = arg_parser.add_argument_group("All Args" if all_args else f"{default_goal} Args")
    inputs = goal_manager.inputs(goal=None if all_args else default_goal, scope=scope)
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
