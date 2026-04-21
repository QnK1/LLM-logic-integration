from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.system_prompts.system_prompts import SystemPrompt
import json


class CriticAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite", system_prompt: str = SystemPrompt.NLTK_CRITIC_PROMPT.value):
        self.model = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0
        )

        self.system_prompt = system_prompt

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", (
                "ORIGINAL SOURCE TEXT: {original_sentence}\n\n"
                "GENERATOR'S PROPOSAL (FOL): {generator_logic}\n\n"
                "SOLVER STATUS (NLTK): {solver_status}"
            ))
        ])

        self.chain = self.prompt | self.model | JsonOutputParser()

    def verify_logic(self, original_sentence: str, generator_output: dict, solver_status: dict) -> dict:
        generator_logic_str = json.dumps(generator_output, indent=2)
        solver_status_str = json.dumps(solver_status, indent=2)

        return self.chain.invoke({
            "original_sentence": original_sentence,
            "generator_logic": generator_logic_str,
            "solver_status": solver_status_str
        })
