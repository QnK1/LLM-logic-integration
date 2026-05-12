from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class DebateAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.3)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.DEBATE_PROMPT.value),
                ("human", "{input_sentence}"),
            ]
        )
        self.chain = self.prompt | self.model | JsonOutputParser()

    def create_prompt(
        self,
        input_sentence: str,
        previous_discussion: str | None = None,
    ) -> dict:
        if previous_discussion is not None:
            refinement_input = (
                f"Original text: {input_sentence}\n"
                f"Previous discusstion: {previous_discussion}"
            )
            return self.chain.invoke({"input_sentence": refinement_input})
        return self.chain.invoke({"input_sentence": input_sentence})
