import io
from collections import deque
from typing import override

from loguru import logger

from llm_logic_integration.agents.arbiter_agent import ArbiterAgent
from llm_logic_integration.agents.debate_agent import DebateAgent
from llm_logic_integration.multi_agent_systems.multi_agent_system import (
    MultiAgentSystem,
)


class DebateSystem(MultiAgentSystem):
    def __init__(
        self,
        debate_agents: list[DebateAgent],
        arbiter_agent: ArbiterAgent,
        iterations: int,
        buffer_size: int,
    ) -> None:
        super().__init__(iterations)
        self.debate_agents = debate_agents
        self.arbiter = arbiter_agent

        self.buffer = deque(maxlen=buffer_size)

    @override
    def run(self, sentence: str) -> str:
        for i in range(self.max_iterations):
            logger.info(f"[Debate System] --- Iteration {i + 1} ---")

            for agent_i, agent in enumerate(self.debate_agents):
                prev_discussion = self._get_prev_discussion()

                out = agent.create_prompt(sentence, prev_discussion)
                answer = out.get("answer", "")
                logger.info(f"Debate agent {agent_i} answer: {answer}")

                self.buffer.append(answer)

        prev_discussion = {"discussion": self._get_prev_discussion()}
        return self.arbiter.decide(sentence, prev_discussion).final_answer

    def _get_prev_discussion(self) -> str | None:
        sb = io.StringIO()

        prev_discussion = None
        if self.buffer:
            sb.truncate(0)
            sb.seek(0)
            for a, ans in enumerate(self.buffer):
                sb.write(f"Answer {a}: {ans}\n")
            prev_discussion = sb.getvalue()

        return prev_discussion
