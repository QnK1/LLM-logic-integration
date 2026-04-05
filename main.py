import os
from dotenv import load_dotenv

from agents.generator_agent import GeneratorAgent
from solvers.nltk_solver import NLTKSolver

from system_prompts.system_prompts import SystemPrompt

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

sentence = "Every human is mortal. Ted is human. Is Ted mortal?"

generator_agent = GeneratorAgent(api_key, model="gemini-2.5-flash")
agent_response = generator_agent.create_prompt(sentence)
print(agent_response)

premises = agent_response["premises"]
goal = agent_response["goal"]

solver = NLTKSolver()
solver.set_premises(premises)
solver.set_goal(goal)
print(solver.prove_goal())
