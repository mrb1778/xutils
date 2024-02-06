from typing import Dict, Any, Optional


class ExpressionEvaluator:
    def is_expression(self, expression) -> bool:
        raise NotImplementedError

    def eval(self, expression, variable_resolver):
        raise NotImplementedError

class DictVariableResolver:
    def __init__(self, data: Dict):
        super().__init__()
        self.data: Dict = data

    def __call__(self, key: str) -> Optional[Any]:
        return self.data[key] if key in self.data else key
