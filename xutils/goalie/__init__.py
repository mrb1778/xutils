import types

from .from_args import run_from_arg_parse
from .goal import GoalManager, MissingGoal, GoalParam
from .std_lib import init_lib

# GoalManager.run_from_arg_parse = types.MethodType(run_from_arg_parse, GoalManager)

goal = GoalManager()
init_lib(goal, goal.SCOPE_STANDARD)
