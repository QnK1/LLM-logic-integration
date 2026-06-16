from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class DebateOutput(BaseModel):
    answer: str = Field(
        description="Your natural language answer, with a brief description of reasoning used."
    )


class DebateAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.5)

        self.structured_model = self.model.with_structured_output(DebateOutput)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.DEBATE_PROMPT.value),
                ("human", "{input_sentence}"),
            ]
        )
        self.chain = self.prompt | self.structured_model

    def create_prompt(
        self,
        input_sentence: str,
        previous_discussion: str | None = None,
    ) -> DebateOutput:
        if previous_discussion is not None:
            refinement_input = (
                f"Original text: {input_sentence}\n"
                f"Previous discussion: {previous_discussion}"
            )
            return self.chain.invoke({"input_sentence": refinement_input})  # ty:ignore[invalid-return-type]
        return self.chain.invoke({"input_sentence": input_sentence})  # ty:ignore[invalid-return-type]
