import os
from dotenv import load_dotenv

from agents.generator_agent import GeneratorAgent

from nltk import TableauProver
from nltk.sem import Expression
import nltk

nltk.download("punkt")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

sentence = "Every human is good except for Ted"

generator_agent = GeneratorAgent(api_key, model="gemini-2.5-flash")
result = generator_agent.create_prompt(sentence)

read_expr = Expression.fromstring
expr = read_expr(result)
premises = [expr]
goal = read_expr("-good(ted)")

prover = TableauProver()
result = prover.prove(goal, premises)
print(result)

