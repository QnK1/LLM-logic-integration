import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nltk import TableauProver

from nltk.sem import Expression
import nltk

nltk.download("punkt")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0
)

system_prompt = """
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

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input_sentence}")
])

chain = prompt | model | StrOutputParser()

sentence = "Every human is good except for Ted"
result = chain.invoke({"input_sentence": sentence})

print(f"Sentence: {sentence}")
print(f"Logic: {result}")

read_expr = Expression.fromstring
expr = read_expr(result)
premises = [expr]
goal = read_expr("-good(ted)")

prover = TableauProver()
result = prover.prove(goal, premises)
print(result)
