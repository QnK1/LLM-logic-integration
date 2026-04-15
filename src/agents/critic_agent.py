from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.system_prompts.system_prompts import SystemPrompt
import json

class CriticAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite", system_prompt: str = SystemPrompt.CRITIC_PROMPT.value):
        self.model = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0
        )

        self.system_prompt = system_prompt

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "ORIGINAL SENTENCE: {original_sentence}\nGENERATED LOGIC: {generated_logic}")
        ])

        self.chain = self.prompt | self.model | JsonOutputParser()

    def verify_logic(self, original_sentence: str, generated_logic: dict) -> dict:
        logic_string = json.dumps(generated_logic, indent=2)
        return self.chain.invoke({"original_sentence": original_sentence, "generated_logic": logic_string})
