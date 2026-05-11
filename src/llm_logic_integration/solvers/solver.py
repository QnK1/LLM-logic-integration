from abc import ABC, abstractmethod


class Solver(ABC):
    @abstractmethod
    def set_premises(self, logic_formulas: list) -> None:
        pass

    @abstractmethod
    def set_goal(self, logic_formula) -> None:
        pass

    @abstractmethod
    def prove_goal(self) -> bool:
        pass

    @abstractmethod
    def return_status(self) -> dict:
        pass
