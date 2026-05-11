import z3

from llm_logic_integration.solvers.solver import Solver


class Z3Solver(Solver):
    def __init__(self):
        self.solver = z3.Solver()
        self.premises: list[z3.BoolRef] = []
        self.goal: z3.BoolRef | None = None
        self.last_result: z3.CheckSatResult | None = None

    def set_premises(self, logic_formulas: list) -> None:
        self.solver.reset()
        self.premises = logic_formulas
        for i, formula in enumerate(logic_formulas):
            label = z3.Bool(f"p{i}")
            self.solver.assert_and_track(formula, label)

    def set_goal(self, logic_formula: z3.BoolRef) -> None:
        self.goal = logic_formula

    def prove_goal(self) -> bool:
        if self.goal is None:
            raise ValueError("Goal has not been set.")
        self.solver.push()
        goal_negation = z3.Not(self.goal)
        self.solver.assert_and_track(goal_negation, "goal_negation")
        self.last_result = self.solver.check()
        is_valid = self.last_result == z3.unsat
        if not is_valid:
            self.solver.pop()
        return is_valid

    def return_status(self) -> dict:
        if self.last_result == z3.unsat:
            core = self.solver.unsat_core()
            core_labels = [str(c) for c in core]
            if "goal_negation" in core_labels:
                return {"type": "SUCCESS", "message": "The proof is logically valid."}
            else:
                return {"type": "UNSAT_PREMISES", "message": "Inconsistent premises."}
        elif self.last_result == z3.sat:
            return {"type": "COUNTERMODEL", "counterexample": str(self.solver.model())}
        return {"type": "UNKNOWN"}
