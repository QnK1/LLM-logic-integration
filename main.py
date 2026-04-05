import os
from dotenv import load_dotenv

from agents.critic_agent import CriticAgent
from agents.generator_agent import GeneratorAgent
from multi_agent_systems.simple_multi_agent_system import SimpleMultiAgentSystem
from solvers.nltk_solver import NLTKSolver

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

sentence = "Every human is mortal. Ted is human. Is Ted mortal?"

generator_agent = GeneratorAgent(api_key, model="gemini-2.5-flash")
critic_agent = CriticAgent(api_key, model="gemini-2.5-flash")
solver = NLTKSolver()

simple_mas = SimpleMultiAgentSystem(5, generator_agent, critic_agent, solver)
print(simple_mas.run(sentence))
