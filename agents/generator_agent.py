from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class GeneratorAgent:
    def __init__(self, api_key, model="gemini-2.5-flash-lite"):
        self.model = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0
        )

        self.system_prompt = """
        You are a logic translator for NLTK.
        Convert sentences to First-Order Logic using EXACTLY this syntax:
        - Predicates: human(x), mortal(x) (lowercase, no spaces before parenthesis)
        - Quantifiers: 'all x.' for ∀ and 'exists x.' for ∃
        - Connectives:
            - '&' for AND
            - '|' for OR
            - '->' for IMPLIES
            - '-' for NOT
        - NO question marks before variables (use 'x' not '?x')
        - NO LISP-style prefix notation (use 'A & B' not 'AND A B')

        Example:
        Sentence: All humans are mortal and Socrates is not a bird.
        Logic: all x.(human(x) -> mortal(x)) & -bird(socrates)
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input_sentence}")
        ])

        self.chain = self.prompt | self.model | StrOutputParser()

    def create_prompt(self, input_sentence: str):
        return self.chain.invoke({"input_sentence": input_sentence})