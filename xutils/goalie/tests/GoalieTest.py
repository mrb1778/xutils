import unittest

from xutils.goalie import goal, MissingGoal, GoalParam


class GoalieTest(unittest.TestCase):
    def test_basic(self):
        goal.reset()

        @goal
        def test_fun():
            return 1

        self.assertEqual(1, test_fun())
        self.assertEqual(1, goal.run(test_fun))
        self.assertEqual(1, goal.run("test_fun"))

    def test_basic_params(self):
        goal.reset()

        @goal
        def test_fun(x, y):
            return x + y

        self.assertEqual(3, test_fun(x=1, y=2))
        self.assertEqual(3, goal.run(test_fun, x=1, y=2))
        self.assertEqual(3, goal.run("test_fun", x=1, y=2))

    def test_pre(self):
        goal.reset()

        other_called = False

        @goal
        def pre_fun():
            nonlocal other_called
            other_called = True
            return 1

        @goal(pre=pre_fun)
        def my_fun():
            return 2

        self.assertEqual(2, my_fun())
        self.assertEqual(other_called, True)
        other_called = False

        self.assertEqual(2, goal.run(my_fun))
        self.assertEqual(other_called, True)
        other_called = False

        self.assertEqual(2, goal.run("my_fun"))
        self.assertEqual(other_called, True)

    def test_goal_params(self):
        goal.reset()

        @goal
        def my_fun():
            return 1

        self.assertEqual(1, my_fun())

        @goal(params={"x": my_fun.result()})
        def my_fun_2(x):
            return x + 1

        self.assertEqual(2, my_fun_2())

        @goal
        def my_fun_3(y=my_fun.result()):
            return y + 1

        self.assertEqual(2, my_fun_3())

    # def test_default_scope(self):
    #     goal.reset()
    #
    #     with goal.in_scope("test"):
    #         @goal
    #         def my_fun():
    #             return 1
    #
    #         self.assertEqual(1, my_fun())
    #
    #         @goal(params={"x": my_fun.result()})
    #         def my_fun_2(x):
    #             return x + 1
    #
    #         self.assertEqual(2, my_fun_2())
    #
    #         @goal
    #         def my_fun_3(y=my_fun.result()):
    #             return y + 1
    #
    #         self.assertEqual(2, my_fun_3())
    #
    #     self.assertEqual(2, goal.run(scope="test", name="my_fun_2"))

    def test_goal_pass(self):
        goal.reset()

        @goal
        def my_fun(x=0, y=0):
            return x + y

        self.assertEqual(3, my_fun(x=1, y=2))

        @goal
        def my_fun_2(x=0,
                     y=0,
                     f=my_fun):
            return f(x=x, y=y)

        self.assertEqual(5, my_fun_2(x=1, y=4))
        self.assertEqual(5, goal.run(my_fun_2, x=1, y=4))

    def test_goal_pass_chain(self):
        goal.reset()

        @goal
        def my_fun_pre_num(in_num: int = 0):
            return in_num + 5

        @goal
        def my_fun_pre(num: int = my_fun_pre_num.result()):
            return num

        @goal
        def my_fun(x=0, y=0, z=my_fun_pre.result()):
            return x + y + z

        self.assertEqual(13, my_fun(x=1, y=2, in_num=5))

        @goal
        def my_fun_2(x=0,
                     y=0,
                     f=my_fun):
            return f(x=x, y=y)

        self.assertEqual(15, my_fun_2(x=1, y=4, in_num=5))
        self.assertEqual(15, goal.run(my_fun_2, x=1, y=4, in_num=5))

    def test_params(self):
        goal.reset()

        @goal
        def my_fun():
            return 3

        @goal
        def my_fun_2(bob, sally=my_fun.result()):
            return bob * sally

        self.assertEqual(my_fun(), 3)
        self.assertEqual(my_fun_2(bob=2), 6)
        self.assertEqual(goal.run("my_fun_2", bob=2), 6)
        # self.assertEqual(goal.requirements("my_fun_2"), {"my_fun", "my_fun_2"})
        self.assertEqual([input_ for input_ in goal.inputs("my_fun_2").keys()], ['bob'])

    def test_with_fun_result(self):
        goal.reset()

        @goal
        def my_fun():
            return 3

        @goal(params={"factor": my_fun.result()})
        def my_fun_2(factor: int):
            return 10 * factor

        self.assertEqual(goal.run("my_fun_2"), 30)

    def test_with_fun_ref(self):
        goal.reset()

        @goal
        def my_fun():
            return 3

        @goal
        def my_fun_2(goal_ref=my_fun):
            return 10 * goal_ref()

        self.assertEqual(my_fun_2(), 30)
        self.assertEqual(goal.run("my_fun_2"), 30)

    def test_with_fun_ref_mixed(self):
        goal.reset()

        @goal
        def my_fun_2():
            return 2

        @goal
        def my_fun_3():
            return 3

        @goal
        def my_fun_4(goal_ref=my_fun_2, goal_ref_2=my_fun_3.result()):
            return 4 * goal_ref() * goal_ref_2

        @goal
        def my_fun_5(goal_ref_2=my_fun_3.result(), goal_ref=my_fun_2):
            return 5 * goal_ref() * goal_ref_2

        self.assertEqual(my_fun_2(), 2)
        self.assertEqual(my_fun_3(), 3)

        self.assertEqual(my_fun_4(), 2*3*4)
        self.assertEqual(goal.run("my_fun_4"), 2*3*4)

        self.assertEqual(my_fun_5(), 2*3*5)
        self.assertEqual(goal.run("my_fun_5"), 2*3*5)

    def test_args(self):
        goal.reset()

        @goal
        def my_fun():
            return 5

        @goal(params={"factor": my_fun.result()})
        def my_fun_2(factor):
            return 100 * factor

        self.assertEqual(my_fun_2(factor=5), 500)
        self.assertEqual(my_fun_2(), 500)

    def test_input_params(self):
        goal.reset()

        @goal
        def my_fun(multiplier=5):
            return 5 * multiplier

        @goal(params={"factor": my_fun.result()})
        def my_fun_2(factor):
            return 100 * factor

        self.assertEqual(goal.run("my_fun_2"), 2500)
        self.assertEqual(goal.run("my_fun_2", multiplier=10), 5000)

    def test_goal_param_args(self):
        goal.reset()

        @goal
        def base_fun(multiplier: int = 5) -> int:
            return 5 * multiplier

        self.assertEqual(goal.run("base_fun", multiplier=2), 10)
        self.assertEqual(goal.run("base_fun"), 25)

    def test_cross_scope(self):
        goal.reset()

        @goal(scope="standard")
        def standard_fun(multiplier=5):
            return 5 * multiplier

        @goal(scope="different", params={"factor": standard_fun.result(multiplier=100)})
        def my_fun_2(factor):
            return 100 * factor

        @goal(params={"factor": my_fun_2.result()})
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

        @goal(name="x_my_fun_2", params={"factor": my_fun.result()})
        def my_fun_2(factor):
            return 100 * factor

        @goal(name="x_my_fun_3", params={"factor": goal.get("x_my_fun").result()})
        def my_fun_3(factor):
            return 1000 * factor

        self.assertEqual(goal.run("x_my_fun_2"), 500)
        self.assertEqual(my_fun_2(), goal.run("x_my_fun_2"))
        self.assertEqual(goal.run("x_my_fun_3"), 5000)
        self.assertEqual(my_fun_3(), goal.run("x_my_fun_3"))

    def test_fun_params(self):
        goal.reset()

        @goal
        def my_fun():
            return 5

        @goal
        def my_fun_2(factor: int = my_fun.result()):
            return 100 * factor

        self.assertEqual(goal.run(my_fun_2), 100 * 5)

        @goal(params={"factor2": my_fun.result()})
        def my_fun_3(factor: int = my_fun.result(), factor2=None):
            return 100 * factor * factor2

        self.assertEqual(goal.run(my_fun_3), 100 * 5 * 5)
        # self.assertEqual(my_fun_2(), goal.run("x_my_fun_2"))
        # self.assertEqual(goal.run("x_my_fun_3"), 5000)
        # self.assertEqual(my_fun_3(), goal.run("x_my_fun_3"))    def test_manual_name(self):
        # goal.reset()

        @goal(name="x_my_fun")
        def my_fun():
            return 5

        @goal(name="x_my_fun_2", params={"factor": my_fun.result()})
        def my_fun_2(factor):
            return 100 * factor

        @goal(name="x_my_fun_3", params={"factor": goal.get("x_my_fun").result()})
        def my_fun_3(factor):
            return 1000 * factor

        self.assertEqual(goal.run("x_my_fun_2"), 500)
        self.assertEqual(my_fun_2(), goal.run("x_my_fun_2"))
        self.assertEqual(goal.run("x_my_fun_3"), 5000)
        self.assertEqual(my_fun_3(), goal.run("x_my_fun_3"))

    def test_fun_parse(self):
        pass

    def test_positional(self):
        goal.reset()
        @goal
        def my_fun(factor):
            return 100 * factor

        @goal
        def my_fun_args(factor, *others):
            result = factor
            for x in others:
                result = result * x
            return result

        self.assertEqual(goal.run("my_fun_args", factor=3, others=[2, 5]), 30)
