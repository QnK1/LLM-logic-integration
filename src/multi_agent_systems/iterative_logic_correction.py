from src.agents.critic_agent import CriticAgent
from src.agents.generator_agent import GeneratorAgent
from src.multi_agent_systems.multi_agent_system import MultiAgentSystem
from src.solvers.solver import Solver
from src.utils.mas_logger import mas_logger


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
        original_sentence = sentence
        generator_input = sentence
        for i in range(self.max_iterations):
            generator_output = self.generator_agent.create_prompt(generator_input)
            if self.verbose:
                mas_logger.info(f"Generator output in iteration {i + 1}: {generator_output}")

            premises = generator_output["premises"]
            goal = generator_output["goal"]

            solver_status = self.solver.return_status(premises, goal)

            critic_response = self.critic_agent.verify_logic(original_sentence, generator_output, solver_status)

            if self.verbose:
                mas_logger.info(
                    f"MAS finished the task after {i + 1} iterations, and determined the following result: {result}")

            return result

        if self.verbose:
            mas_logger.info(
                f"Maximum number of iterations ({self.max_iterations}) exceeded, MAS reached no conclusions.")

        return None
