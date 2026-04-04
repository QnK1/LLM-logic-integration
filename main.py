import os
from dotenv import load_dotenv

from agents.generator_agent import GeneratorAgent
from solvers.nltk_solver import NLTKSolver

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

sentence = "Every human is good except for Ted"

generator_agent = GeneratorAgent(api_key, model="gemini-2.5-flash")
agent_parsed_sentence = generator_agent.create_prompt(sentence)
goal = "-good(ted)"
print(agent_parsed_sentence)

solver = NLTKSolver()
solver.set_premises(agent_parsed_sentence)
solver.set_goal(goal)
print(solver.prove_goal())
