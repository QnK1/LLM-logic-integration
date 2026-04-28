from loguru import logger

from llm_logic_integration.agents.arbiter_agent import ArbiterAgent
from llm_logic_integration.agents.critic_agent import CriticAgent
from llm_logic_integration.agents.generator_agent import GeneratorAgent
from llm_logic_integration.multi_agent_systems.multi_agent_system import (
    MultiAgentSystem,
)


class MASNoLogic(MultiAgentSystem):
    def __init__(
        self,
        max_iterations: int,
        generator: GeneratorAgent,
        critic: CriticAgent,
        arbiter: ArbiterAgent,
    ):
        super().__init__(max_iterations)
        self.generator = generator
        self.critic = critic
        self.arbiter = arbiter

    def run(self, sentence: str) -> dict:
        feedback = None
        last_output = None

        for i in range(self.max_iterations):
            logger.info(f"[No Logic MAS] --- Iteration {i + 1} ---")

            gen_out = self.generator.create_prompt(sentence, feedback, last_output)
            gen_answer = gen_out.get("answer", "")
            logger.info(f"Generator answer: {gen_answer}")

            critic_out = self.critic.evaluate(sentence, gen_answer)
            logger.info(f"Critic status: {critic_out['status']}")

            if critic_out["status"] == "OK":
                return self.arbiter.decide(sentence, gen_out)

            logger.warning(f"Critic rejected: {critic_out['reasoning']}")
            feedback = critic_out["feedback"]
            last_output = gen_out

        logger.error("Max iterations reached.")
        return {"final_answer": "FAILED_TO_RESOLVE"}
