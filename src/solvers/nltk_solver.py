import nltk
from nltk import TableauProver
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
