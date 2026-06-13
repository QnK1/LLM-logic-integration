# mas_no_logic.py
from typing import override

from loguru import logger
from pydantic import BaseModel

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
        force_max_iter: bool = False,
    ):
        super().__init__(max_iterations)
        self.generator = generator
        self.critic = critic
        self.arbiter = arbiter
        self.force_max_iter = force_max_iter

    @override
    def run(
        self,
        sentence: str,
        output_schema: type[BaseModel],
    ) -> BaseModel:
        feedback = None
        last_output = None

        for i in range(self.max_iterations):
            logger.info(f"[No Logic MAS] --- Iteration {i + 1} ---")

            gen_out = self.generator.create_prompt(sentence, feedback, last_output)
            gen_answer = gen_out.answer
            logger.info(f"Generator answer: {gen_answer}")

            critic_out = self.critic.evaluate(
                sentence,
                gen_answer,
                verifier_answer="There is only the generator's answer.",
            )
            logger.info(f"Critic status: {critic_out.status}")

            if critic_out.status == "OK":
                if not self.force_max_iter:
                    return self.arbiter.decide(
                        sentence, gen_out.model_dump(), output_schema
                    )
                logger.info("Critic accepted, but force_max_iter is True. Continuing.")

            if critic_out.status != "OK":
                logger.warning(f"Critic rejected: {critic_out.reasoning}")

            feedback = critic_out.feedback
            last_output = gen_out

        logger.error("Max iterations reached.")

        arbiter_input = {
            "generator_answer": gen_answer,
            "critique": critic_out.feedback,
        }
        return self.arbiter.decide(sentence, arbiter_input, output_schema)
