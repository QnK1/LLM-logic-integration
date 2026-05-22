from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from llm_logic_integration.agents.travel_planning.model import TravelPlan
from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class TravelPlannerAgent:
    def __init__(
        self,
        provider: str,
        model_name: str,
        tools: list[BaseTool],
        api_key: str | None = None,
    ):
        self.model = create_llm(provider, model_name, api_key, temperature=0.0)
        self.tools = tools

        self.structured_model = self.model.with_structured_output(TravelPlan)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.TRAVEL_PLANNER_PROMPT.value),
                ("human", "{input_sentence}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def generate_plan(self, input_sentence: str) -> TravelPlan:
        agent_result = self.agent_executor.invoke({"input_sentence": input_sentence})

        raw_text_answer = agent_result["output"]

        formatting_prompt = (
            f"Convert this travel itinerary into the strict TravelPlan schema:\n\n"
            f"{raw_text_answer}"
        )
        final_structured_plan = self.structured_model.invoke(formatting_prompt)

        return final_structured_plan  # ty: ignore[invalid-return-type]
