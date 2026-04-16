from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.system_prompts.system_prompts import SystemPrompt


class GeneratorAgent:
    """
    Agent responsible for translating natural language sentences into formal logical representations.

    It utilizes a Large Language Model (LLM) to parse linguistic input and output a structured
    JSON format containing premises and a goal, following specific logic syntax rules.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite",
                 system_prompt: str = SystemPrompt.NLTK_GENERATOR_PROMPT.value):
        """
        Initializes the GeneratorAgent with necessary API credentials and model configuration.

        Args:
            api_key (str): The API key for authentication.
            model (str): The specific LLM model version to use.
            system_prompt (str): Instructions defining the logic translation rules and output format.
        """
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
        """
        Invokes the LLM chain to translate a natural language sentence into a logical JSON structure.

        Args:
            input_sentence (str): The natural language text containing facts and a conclusion.

        Returns:
            dict: A dictionary containing 'premises' (list of strings) and 'goal' (string).

        Example return:
            {"premises": ["all x.(human(x) -> mortal(x))", "human(socrates)"], "goal": "mortal(socrates)"}
        """
        return self.chain.invoke({"input_sentence": input_sentence})
