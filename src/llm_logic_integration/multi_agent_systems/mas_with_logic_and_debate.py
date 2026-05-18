from llm_logic_integration.agents.arbiter_agent import ArbiterAgent
from llm_logic_integration.multi_agent_systems.debate import DebateSystem
from llm_logic_integration.multi_agent_systems.mas_with_logic import MASWithLogic
from llm_logic_integration.multi_agent_systems.multi_agent_system import (
    MultiAgentSystem,
)


class MASWithLogicAndDebate(MultiAgentSystem):
    def __init__(
        self,
        logic_mas: MASWithLogic,
        debate_system: DebateSystem,
        arbiter_agent: ArbiterAgent,
    ) -> None:
        self.logic_mas = logic_mas
        self.debate_system = debate_system
        self.arbiter = arbiter_agent

    def run(self, sentence: str) -> str:
        logic_mas_answer = self.logic_mas.run(sentence)

        debate_in = (
            "Work out an answer based on the user's prompt and a logic-based system's answer.\n"
            f"User Prompt: {sentence}\n"
            f"Logic System Answer: {logic_mas_answer}\n"
        )
        debate_out = self.debate_system.run(debate_in)

        return self.arbiter.decide(
            sentence,
            {
                "logic_system_answer": logic_mas_answer,
                "debate_result": debate_out,
            },
        ).final_answer
