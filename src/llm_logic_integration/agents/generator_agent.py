from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class GeneratorOutput(BaseModel):
    answer: str = Field(description="Your natural language answer.")


class GeneratorAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.4)

        self.structured_model = self.model.with_structured_output(GeneratorOutput)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.GENERATOR_PROMPT.value),
                ("human", "{input_sentence}"),
            ]
        )
        self.chain = self.prompt | self.structured_model

    def create_prompt(
        self, input_sentence: str, feedback: str | None = None, previous_output=None
    ) -> GeneratorOutput:
        if feedback:
            input_sentence = (
                f"Original text: {input_sentence}\n"
                f"Previous output: {previous_output}\n"
                f"Feedback to fix: {feedback}"
            )
        return self.chain.invoke({"input_sentence": input_sentence})  # ty:ignore[invalid-return-type]
