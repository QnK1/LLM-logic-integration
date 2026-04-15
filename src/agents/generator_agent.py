from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.system_prompts.system_prompts import SystemPrompt


class GeneratorAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite", system_prompt: str = SystemPrompt.NLTK_GENERATOR_PROMPT.value):
        self.model = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0
        )

        self.system_prompt = system_prompt

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input_sentence}")
        ])

        self.chain = self.prompt | self.model | JsonOutputParser()

    def create_prompt(self, input_sentence: str) -> dict:
        return self.chain.invoke({"input_sentence": input_sentence})
