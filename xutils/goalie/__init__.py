from .goal import GoalManager, MissingGoal, GoalParam
from .std_lib import init_lib

goal = GoalManager()
init_lib(goal, goal.SCOPE_STANDARD)
