import os
from dotenv import load_dotenv

from src.agents.critic_agent import CriticAgent
from src.agents.generator_agent import GeneratorAgent
from src.multi_agent_systems.iterative_logic_correction import IterativeLogicCorrectionMAS
from src.multi_agent_systems.simple_multi_agent_system import SimpleMultiAgentSystem
from src.solvers.nltk_solver import NLTKSolver

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

sentence = """
Anyone who was in the Garden and is not a Guard must be a Guest.
If the Butler is a Guest, then the Butler is not the Assassin.
If the Butler was in the Garden, then he is not a Guard.
Everyone is either the Assassin or they are Innocent.
The Butler was in the Garden.
All Guests are Innocent.
Is the Butler innocent?
"""

generator_agent = GeneratorAgent(api_key, model="gemini-2.5-flash-lite")
critic_agent = CriticAgent(api_key, model="gemini-2.5-flash-lite")
solver = NLTKSolver()

# simple_mas = SimpleMultiAgentSystem(5, generator_agent, critic_agent, solver)
# print(simple_mas.run(sentence))

iter_logic_corr = IterativeLogicCorrectionMAS(5, generator_agent, critic_agent, solver)
print(iter_logic_corr.run(sentence))
