from abc import ABC, abstractmethod


class Solver(ABC):
    @abstractmethod
    def set_premises(self, logic_formulas: list[str]) -> None:
        pass

    @abstractmethod
    def set_goal(self, logic_formula: str) -> None:
        pass

    @abstractmethod
    def prove_goal(self) -> bool:
        pass

    @abstractmethod
    def return_status(self, premises: list[str], goal: str) -> dict:
        pass
