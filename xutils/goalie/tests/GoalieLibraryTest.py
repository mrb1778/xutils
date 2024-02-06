import unittest

from xutils.goalie import goal


class GoalieTest(unittest.TestCase):
    def test_basic(self):
        kwargs = {
            "these": [1, 2, 3],
            "how": lambda x, y: x + y,
            "initial": 0
        }
        self.assertEqual(goal.lib.std.reduce(**kwargs), 6)
        self.assertEqual(goal.run("reduce", scope="std", **kwargs), 6)

        # @goal
        # def loop_body_x(x):
        #     return x + 1

        # @goal
        # def call_join(xs, joiner=goal.lib.std.join):
        #     return joiner(over=xs, how=lambda x, y: x + y,  initial_value=0)
        #
        # self.assertEqual(call_join(x=[1, 2, 3]), 5)


