import re

import nltk
from nltk import TableauProver, LogicalExpressionException
from nltk.sem import Expression
from src.solvers.solver import Solver


nltk.download("punkt")

class NLTKSolver(Solver):
    def __init__(self):
        self.read_expr = Expression.fromstring
        self.prover = TableauProver()
        self.premises = None
        self.goal = None

    def set_premises(self, logic_formulas: list[str]) -> None:
        self.premises = [self.read_expr(formula) for formula in logic_formulas]

    def set_goal(self, logic_formula: str) -> None:
        self.goal = self.read_expr(logic_formula)

    def prove_goal(self) -> bool:
        if not self.premises or not self.goal:
            raise AttributeError("Both premises and goal must be defined before proving.")

        return self.prover.prove(self.goal, self.premises)

    def return_status(self, premises: list[str], goal: str) -> dict:
        try:
            self.set_premises(premises)
            self.set_goal(goal)

            all_logic_text = " ".join(premises + [goal])
            atoms = set(re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', all_logic_text))
            keywords = {'all', 'exists', 'and', 'or', 'not', 'implies', 'iff'}
            found_entities = list(atoms - keywords)

            is_true = self.prove_goal()

            neg_goal = self.read_expr(f"-({goal})")
            is_false = self.prover.prove(neg_goal, self.premises)

            if is_true:
                result = "TRUE"
            elif is_false:
                result = "FALSE"
            else:
                result = "UNKNOWN"

            return {
                "status": "SUCCESS",
                "result": result,
                "entities": found_entities,
                "is_consistent": not self.prover.prove(self.read_expr("False"), self.premises)
            }

        except LogicalExpressionException as e:
            return {
                "status": "FAILURE",
                "error_type": "SYNTAX_ERROR",
                "message": str(e)
            }
        except Exception as e:
            return {
                "status": "FAILURE",
                "error_type": "RUNTIME_ERROR",
                "message": str(e)
            }
