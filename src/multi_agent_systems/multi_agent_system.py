from abc import ABC, abstractmethod


class MultiAgentSystem(ABC):
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    @abstractmethod
    def run(self, sentence: str) -> bool:
        pass