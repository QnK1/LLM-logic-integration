from src.agents.critic_agent import CriticAgent
from src.agents.generator_agent import GeneratorAgent
from src.multi_agent_systems.multi_agent_system import MultiAgentSystem
from src.solvers.solver import Solver


class IterativeLogicCorrectionMAS(MultiAgentSystem):
    def __init__(self, max_iterations: int,
                 generator_agent: GeneratorAgent,
                 critic_agent: CriticAgent,
                 solver: Solver,
                 verbose: bool = True):

        super().__init__(max_iterations)
        self.generator_agent = generator_agent
        self.critic_agent = critic_agent
        self.solver = solver
        self.verbose = verbose

    def run(self, sentence: str) -> bool:
        pass
