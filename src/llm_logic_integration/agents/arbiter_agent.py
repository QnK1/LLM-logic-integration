import json

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class ArbiterAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.0)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.ARBITER_PROMPT.value),
                (
                    "human",
                    "ORIGINAL TEXT: {original_sentence}\nAPPROVED ANSWER DATA: {approved_data}",
                ),
            ]
        )
        self.chain = self.prompt | self.model | JsonOutputParser()

    def decide(self, original_sentence: str, approved_data: dict) -> dict:
        return self.chain.invoke(
            {
                "original_sentence": original_sentence,
                "approved_data": json.dumps(approved_data, indent=2),
            }
        )
