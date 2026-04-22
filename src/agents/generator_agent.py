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

    def create_prompt(self, input_sentence: str, feedback: str = None, previous_output=None) -> dict:
        """
        Translates natural language into a logical JSON structure, with optional iterative refinement.

        This method invokes the LLM chain to parse input text into formal logic. If feedback from
        a CriticAgent is provided, it performs a refinement step by presenting the previous
        errors and the original context to the model for correction.

        Args:
            input_sentence (str): The original natural language text containing facts and a conclusion.
            feedback (str, optional): Error messages or logical critiques from the Critic/Solver.
                Defaults to None.
            previous_output (dict, optional): The failed logical structure that needs correction.
                Defaults to None.

        Returns:
            dict: A dictionary containing 'premises' (list of strings) and 'goal' (string).

        Example:
            Standard call:
                create_prompt("All men are mortal. Socrates is a man. Is Socrates mortal?")

            Refinement call:
                create_prompt("All men are mortal...", feedback="Syntax error: missing parenthesis",
                              previous_output={"premises": ["all x.human(x -> mortal(x)"], "goal": "..."})
        """

        if feedback:
            refinement_input = (
                f"Your previous output was: {previous_output}\n"
                f"The Critic provided the following feedback: {feedback}\n"
                f"Please correct the logic formulas based on this feedback for the original text: {input_sentence}"
            )
            return self.chain.invoke({"input_sentence": refinement_input})

        return self.chain.invoke({"input_sentence": input_sentence})
