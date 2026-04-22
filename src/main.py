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
1. Every device on the network is either a Server, a Workstation, or a Gateway.
2. All Gateways must have an Active Firewall and a Valid Certificate.
3. Any device that has an Active Firewall and is not Outdated is considered Secure.
4. If a device is Secure, it can host an Encrypted Database.
5. A device that hosts an Encrypted Database and has Admin Access is a Critical Asset.
6. All Critical Assets must be Monitored.
7. If a device is Monitored and is a Server, it is Eligible for Backup.
8. Every device Eligible for Backup that has High Priority is part of the Recovery Plan.
9. If a device is in the Recovery Plan, it is Insurance Compliant.

The Network Topology:
10. Device_A is a Server.
11. Device_A is not Outdated.
12. Device_A has Admin Access.
13. Device_A has High Priority.
14. Device_A has an Active Firewall.
15. Device_B is a Gateway.
16. Device_B has a Valid Certificate.
17. If Device_B is a Gateway, then Device_B has an Active Firewall.

The Security Rules:
18. Any device that is Insurance Compliant is highly protected.
19. If a device is highly protected, it is not a Liability.
20. All devices that are not Liabilities are Operational.

The Goal:
Is Device_A Operational?
"""

generator_agent = GeneratorAgent(api_key, model="gemini-2.5-flash-lite")
critic_agent = CriticAgent(api_key, model="gemini-2.5-flash-lite")
solver = NLTKSolver()

# simple_mas = SimpleMultiAgentSystem(5, generator_agent, critic_agent, solver)
# print(simple_mas.run(sentence))

iter_logic_corr = IterativeLogicCorrectionMAS(5, generator_agent, critic_agent, solver)
print(iter_logic_corr.run(sentence))
