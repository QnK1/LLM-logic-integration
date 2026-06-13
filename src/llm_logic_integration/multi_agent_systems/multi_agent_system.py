# multi_agent_system.py
from abc import ABC, abstractmethod

from pydantic import BaseModel


class MultiAgentSystem(ABC):
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    @abstractmethod
    def run(self, sentence: str, output_schema: type[BaseModel]) -> BaseModel:
        pass
