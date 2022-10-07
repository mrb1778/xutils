import unittest

from xutils.goalie.goal import goal, InputParam, MissingGoal, GoalParam


class GoalieTest(unittest.TestCase):
    def test_basic(self):
        goal.reset()

        other_called = False

        @goal(name="pre_fun")
        def pre_fun():
            nonlocal other_called
            other_called = True
            return 1

        @goal(name="my_fun", pre=pre_fun)
        def my_fun():
            return 2

        self.assertEqual(my_fun(), 2)
        self.assertEqual(other_called, True)
        self.assertEqual(pre_fun(), 1)

    # def test_basic_params(self):
    #     goal.reset()
    #
    #     @goal
    #     def my_fun():
    #         return 2
    #
    #     @goal(params={"result_1": my_fun, "result_2": my_fun})
    #     def my_fun_2(result_1, result_2):
    #         return result_1, result_2
    #
    #     @goal("my_fun_param")
    #     def my_fun_param(start=0):
    #         return start
    #
    #     @goal("my_fun_3", pre=["my_fun_param", "my_fun_param"])
    #     def my_fun_3(results, start=0):
    #         return sum(results, start)
    #
    #     self.assertEqual(goal.run("my_fun_2"), (2, 2))
        # self.assertEqual(my_fun_3(start=3), 9)
        # self.assertEqual(goal.run("my_fun_3", start=3), 9)

    # def test_mapped_goals(self):
    #     @goal
    #     def my_fun():
    #         return 3
    #
    #     @goal(pre={"my_fun": "sally"})
    #     def my_fun_2(bob, sally):
    #         return bob * sally
    #
    #     self.assertEqual(my_fun(), 3)
    #     self.assertEqual(my_fun_2(bob=2), 6)
    #     self.assertEqual(goal.run("my_fun_2", bob=2), 6)
    #
    # def test_mapped_goals_list(self):
    #     @goal
    #     def my_fun():
    #         return 3
    #
    #     @goal(pre=[("my_fun", "sally")])
    #     def my_fun_2(bob=None, sally=None):
    #         return bob * sally
    #
    #     self.assertEqual(my_fun(), 3)
    #     self.assertEqual(my_fun_2(bob=2), 6)
    #     self.assertEqual(goal.run("my_fun_2", bob=2), 6)

    def test_params(self):
        goal.reset()

        @goal
        def my_fun():
            return 3

        @goal(params={"sally": my_fun, "bob": InputParam("Bob")})
        def my_fun_2(bob, sally):
            return bob * sally

        self.assertEqual(my_fun(), 3)
        self.assertEqual(my_fun_2(bob=2), 6)
        self.assertEqual(goal.run("my_fun_2", bob=2), 6)
        self.assertEqual(goal.requirements("my_fun_2"), {"my_fun", "my_fun_2"})
        self.assertEqual([input_.name for input_ in goal.inputs("my_fun_2")], ['Bob'])

    def test_with_fun(self):
        goal.reset()

        @goal
        def my_fun():
            return 3

        @goal(params={"factor": my_fun})
        def my_fun_2(factor: int):
            return 10 * factor

        self.assertEqual(goal.run("my_fun_2"), 30)

    def test_args(self):
        goal.reset()

        @goal
        def my_fun():
            return 5

        @goal(params={"factor": my_fun})
        def my_fun_2(factor):
            return 100 * factor

        self.assertEqual(goal.run("my_fun_2"), 500)

    # def test_other_scope(self):
    #     goal.reset()
    #
    #     @goal
    #     def my_fun():
    #         return 5
    #
    #     @goal(params={"factor": my_fun})
    #     def my_fun_2(factor):
    #         return 100 * factor
    #
    #     self.assertEqual(goal.run("my_fun_2"), 500)
    #
    #     with goal.scope("my_scope"):
    #         self.assertEqual(goal.run("my_fun_2", scope=goal.SCOPE_DEFAULT), 500)
    #
    #         with self.assertRaises(MissingGoal):
    #             goal.run("my_fun_2")
    #
    #         @goal("my_fun_x")
    #         def my_fun():
    #             return 3
    #
    #         with self.assertRaises(MissingGoal):
    #             goal.run("my_fun_x", scope=goal.SCOPE_DEFAULT)
    #
    #         @goal
    #         def my_fun():
    #             return 3
    #
    #         @goal(params={"factor": my_fun})
    #         def my_fun_2(factor):
    #             return 10 * factor
    #
    #         self.assertEqual(goal.run("my_fun_2"), 30)
    #
    #     self.assertEqual(goal.run("my_fun_2", scope="my_scope"), 30)

    def test_input_params(self):
        goal.reset()

        @goal
        def my_fun(multiplier=5):
            return 5 * multiplier

        @goal(params={"factor": my_fun})
        def my_fun_2(factor):
            return 100 * factor

        self.assertEqual(goal.run("my_fun_2"), 2500)
        self.assertEqual(goal.run("my_fun_2", multiplier=10), 5000)

    def test_goal_param_args(self):
        goal.reset()

        @goal
        def my_fun(multiplier=5):
            return 5 * multiplier

        @goal(params={"factor": GoalParam(my_fun, multiplier=100)})
        def my_fun_2(factor):
            return 100 * factor

        self.assertEqual(goal.run("my_fun_2", multiplier=10), 5000)
        self.assertEqual(goal.run("my_fun_2"), 50000)

    def test_cross_scope(self):
        goal.reset()

        @goal(scope="standard")
        def standard_fun(multiplier=5):
            return 5 * multiplier

        @goal(scope="different", params={"factor": GoalParam(standard_fun, multiplier=100)})
        def my_fun_2(factor):
            return 100 * factor

        @goal(params={"factor": my_fun_2})
        def my_fun_3(factor):
            return 1000 * factor

        with self.assertRaises(MissingGoal):
            goal.run("my_fun_2")
        self.assertEqual(goal.run("my_fun_2", scope="different"), 50000)
        self.assertEqual(goal.run(my_fun_2), 50000)
        self.assertEqual(goal.run(my_fun_2, multiplier=10), 5000)
        self.assertEqual(goal.run(my_fun_3, multiplier=10), 5000000)
        self.assertEqual(goal.run("my_fun_3", multiplier=10), 5000000)

    def test_manual_name(self):
        goal.reset()

        @goal(name="x_my_fun")
        def my_fun():
            return 5

        @goal(name="x_my_fun_2", params={"factor": my_fun})
        def my_fun_2(factor):
            return 100 * factor

        @goal(name="x_my_fun_3", params={"factor": "x_my_fun"})
        def my_fun_3(factor):
            return 1000 * factor

        self.assertEqual(goal.run("x_my_fun_2"), 500)
        self.assertEqual(my_fun_2(), goal.run("x_my_fun_2"))
        self.assertEqual(goal.run("x_my_fun_3"), 5000)
        self.assertEqual(my_fun_3(), goal.run("x_my_fun_3"))