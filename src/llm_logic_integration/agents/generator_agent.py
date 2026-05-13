from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class GeneratorAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.5)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.GENERATOR_PROMPT.value),
                ("human", "{input_sentence}"),
            ]
        )
        self.chain = self.prompt | self.model | JsonOutputParser()

    def create_prompt(
        self, input_sentence: str, feedback: str | None = None, previous_output=None
    ) -> dict:
        if feedback:
            refinement_input = (
                f"Original text: {input_sentence}\n"
                f"Previous output: {previous_output}\n"
                f"Feedback to fix: {feedback}"
            )
            return self.chain.invoke({"input_sentence": refinement_input})
        return self.chain.invoke({"input_sentence": input_sentence})
