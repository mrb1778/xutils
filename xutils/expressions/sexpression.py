import re
from typing import Pattern, Any

from .expression_evaluator import ExpressionEvaluator


class SimpleExpressionEvaluator(ExpressionEvaluator):
    expression_start: str
    expression_end: str

    expression_matcher: Pattern

    def __init__(self, expression_regex: str = "\{\{(.*?)\}\}") -> None:
        super().__init__()
        self.expression_regex = expression_regex
        self.expression_matcher: Pattern = re.compile(self.expression_regex)

    # def is_expression(self, expression) -> bool:
    #     return isinstance(expression, str) and \
    #            re.fullmatch(self.expression_matcher, expression)

    def has_expression(self, expression) -> bool:
        return isinstance(expression, str) and \
               re.search(self.expression_matcher, expression)

    def eval(self, expression, variable_resolver) -> Any:
        if isinstance(expression, str):
            result_concat = []
            last_end = 0
            expression_len = len(expression)
            for res in re.finditer(self.expression_matcher, expression):
                start = res.start()
                end = res.end()
                match = res.group()
                if start == 0 and end == expression_len:
                    return self._eval_expression(self._prep_expression(match),
                                                 variable_resolver)
                else:
                    if last_end != start:
                        result_concat.append(expression[last_end: start])
                    result_concat.append(self._eval_expression(self._prep_expression(match),
                                                               variable_resolver))
                    last_end = end
            if len(result_concat) != 0:
                if last_end != expression_len:
                    result_concat.append(expression[last_end: expression_len])
                return ''.join(map(str, result_concat))
            else:
                return expression
        else:
            return expression


        # if self.is_expression(expression):
        #     return self._eval_expression(expression,
        #                                  variable_resolver)
        # elif self.has_expression(expression):
        #     print("sexpression.py::eval:35")
        #     exit()
        #     # return expression
        # else:
        #     return expression

    def _prep_expression(self, expression: str) -> str:
        return re.findall(self.expression_matcher, expression)[0]
        # return expression.replace(self.expression_start, "").replace(self.expression_end, "")

    def _eval_expression(self, expression: str, variable_resolver):
        split_expression = expression.split()
        if len(split_expression) == 1:
            return self._resolve_variable(expression, variable_resolver)
        if len(split_expression) == 3:
            left, oper, right = split_expression
            left = self._resolve_variable(left, variable_resolver)
            right = self._resolve_variable(right, variable_resolver)

            return self._eval_ternary(left, oper, right)
        else:
            raise ValueError("Unsupported Expression must have 1 or 3 parts separated by spaces")

    def _resolve_variable(self, variable, variable_resolver):
        converted_variable, converted = self._convert(variable)
        if converted:
            return converted_variable
        else:
            resolved_variable = variable_resolver(variable)
            # return resolved_variable if resolved_variable else variable
            return resolved_variable if resolved_variable is not None else variable

    def _eval_ternary(self, left, middle, right):
        if middle == "*":
            return left * right
        elif middle == "+":
            return left + right
        elif middle == "-":
            return left - right
        elif middle == "/":
            return left / right
        elif middle == "%":
            return left % right
        elif middle == "**":
            return left ** right
        elif middle == "||":
            return left if left else right
        elif middle == "and" or middle == "&&":
            return left and right
        elif middle == "or" or middle == "||":
            return left or right
        elif middle == ">":
            return left > right
        elif middle == ">=":
            return left >= right
        elif middle == "<":
            return left < right
        elif middle == "<=":
            return left <= right
        elif middle == "==" or middle == "=":
            return left == right
        elif middle == "!=" or middle == "<>":
            return left != right
        else:
            raise ValueError("Unsupported Operator: " + middle)

    def _convert(self, operand):
        converted, successful = self._convert_int(operand)
        if not successful:
            converted, successful = self._convert_float(operand)
        if not successful:
            converted, successful = self._convert_bool(operand)

        return converted, successful

    @staticmethod
    def _convert_float(operand):
        try:
            return float(operand), True
        except ValueError:
            return operand, False

    @staticmethod
    def _convert_int(operand):
        try:
            return int(operand), True
        except ValueError:
            return operand, False

    @staticmethod
    def _convert_bool(operand):
        if operand in [True, 'True', 'true', 't', 'y', 'yes']:
            return True, True
        elif operand in [False, 'False', 'false', 'f', 'n', 'no']:
            return False, True
        else:
            return operand, False
