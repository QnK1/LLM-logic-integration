from agents.critic_agent import CriticAgent
from agents.generator_agent import GeneratorAgent
from multi_agent_systems.multi_agent_system import MultiAgentSystem
from solvers.solver import Solver
from utils.mas_logger import mas_logger

class SimpleMultiAgentSystem(MultiAgentSystem):
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

    def run(self, sentence: str):
        original_sentence = sentence
        generator_input = sentence
        for i in range(self.max_iterations):
            generator_output = self.generator_agent.create_prompt(generator_input)
            if self.verbose:
                mas_logger.info(f"Generator output in iteration {i + 1}: {generator_output}")

            critic_output = self.critic_agent.verify_logic(original_sentence, generator_output)
            if self.verbose:
                mas_logger.info(f"Critic output in iteration {i + 1}: {critic_output}")

            if critic_output["status"] == "ERROR":
                generator_input = (
                    f"Your previous attempt for the sentence: '{original_sentence}' was incorrect. "
                    f"Critic's feedback: {critic_output['feedback']}. "
                    f"Please try again and return the corrected JSON."
                )
                continue

            premises = generator_output["premises"]
            goal = generator_output["goal"]

            self.solver.set_premises(premises)
            self.solver.set_goal(goal)
            result = self.solver.prove_goal()
            if self.verbose:
                mas_logger.info(f"MAS finished the task after {i + 1} iterations, and determined the following result: {result}")

            return result

        if self.verbose:
            mas_logger.info(f"Maximum number of iterations ({self.max_iterations}) exceeded, MAS reached no conclusions.")

        return None
