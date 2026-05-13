from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_logic_integration.system_prompts.system_prompts import SystemPrompt
from llm_logic_integration.utils.llm_factory import create_llm


class CriticAgent:
    def __init__(self, provider: str, model_name: str, api_key: str | None = None):
        self.model = create_llm(provider, model_name, api_key, temperature=0.3)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SystemPrompt.CRITIC_PROMPT.value),
                (
                    "human",
                    "ORIGINAL TEXT: {original_sentence}\nGENERATOR'S ANSWER: {generator_answer}\n LOGIC VERIFIER'S ANSWER: {verifier_answer}",
                ),
            ]
        )
        self.chain = self.prompt | self.model | JsonOutputParser()

    def evaluate(
        self, original_sentence: str, generator_answer: str, verifier_answer: str
    ) -> dict:
        return self.chain.invoke(
            {
                "original_sentence": original_sentence,
                "generator_answer": generator_answer,
                "verifier_answer": verifier_answer,
            }
        )
