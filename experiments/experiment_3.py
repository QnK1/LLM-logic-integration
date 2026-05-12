import os
import sys

from dotenv import load_dotenv
from loguru import logger

from llm_logic_integration.agents.arbiter_agent import ArbiterAgent
from llm_logic_integration.agents.critic_agent import CriticAgent
from llm_logic_integration.agents.debate_agent import DebateAgent
from llm_logic_integration.agents.generator_agent import GeneratorAgent
from llm_logic_integration.agents.logic_verifier_agent import LogicVerifierAgent
from llm_logic_integration.multi_agent_systems.debate import DebateSystem
from llm_logic_integration.multi_agent_systems.mas_with_logic import MASWithLogic
from llm_logic_integration.multi_agent_systems.mas_with_logic_and_debate import (
    MASWithLogicAndDebate,
)
from llm_logic_integration.solvers.z3_solver import Z3Solver

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

PROVIDER = "ollama"
MODEL_NAME = "qwen2.5-coder"


def get_api_key(provider: str) -> str | None:
    provider = provider.lower()
    if provider == "gemini":
        return str(os.getenv("GEMINI_API_KEY"))
    elif provider == "openai":
        return str(os.getenv("OPENAI_API_KEY"))
    return None


def main():
    load_dotenv()
    api_key = str(get_api_key(PROVIDER))

    experiments = {
        "1. Counterfactuals": (
            "All birds are made of liquid metal. Anything made of liquid metal sinks in water. "
            "Penguins are a type of bird. A standard wooden boat floats on water. "
            "Does a penguin float or sink in water?"
        ),
        "2. Deep Multi-Hop Reasoning": (
            "If Alpha is active, then Beta is dormant. If Gamma is unstable, then Delta is active. "
            "If Beta is dormant, then Epsilon is critical. Zeta is always functioning unless Delta is active. "
            "Alpha is active. Gamma is stable. Is Epsilon critical, and is Zeta functioning?"
        ),
        "3. Constraint Satisfaction": (
            "We have four tasks: A, B, C, and D. Task A must be completed before Task C. "
            "Task B cannot be the first or the last task. Task D must be done immediately after Task A. "
            "Can the order be B, A, D, C?"
        ),
        "4. Structural Negation": (
            "If a software system is secure, it must be encrypted. "
            "If a system is encrypted, it uses more processing power. "
            "System X does not use more processing power. Is System X secure?"
        ),
        "5. Parity and Counting": (
            "There are five doors: Red, Blue, Green, Yellow, and Black. Exactly two doors lead to the prize. "
            "If the Red door leads to the prize, the Green door does not. "
            "The Blue door and the Yellow door have the exact same outcome. "
            "The Black door does not lead to the prize. The Green door leads to the prize. "
            "Which other door leads to the prize?"
        ),
    }

    logger.info(f"Initializing agents using {PROVIDER.upper()} model: {MODEL_NAME}")

    generator = GeneratorAgent(
        provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key
    )
    critic = CriticAgent(provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key)
    arbiter = ArbiterAgent(provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key)

    debate_agents = [
        DebateAgent(provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key),
        DebateAgent(provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key),
    ]

    solver = Z3Solver()
    logic_verifier = LogicVerifierAgent(
        provider=PROVIDER,
        model_name=MODEL_NAME,
        solver=solver,
        api_key=api_key,
        max_retries=8,
    )

    system_a_debate = DebateSystem(
        debate_agents=debate_agents,
        arbiter_agent=arbiter,
        iterations=2,
        buffer_size=3,
    )

    system_b_logic = MASWithLogic(
        max_iterations=5,
        generator=generator,
        critic=critic,
        logic_verifier=logic_verifier,
        arbiter=arbiter,
    )

    system_c_combined = MASWithLogicAndDebate(
        logic_mas=system_b_logic,
        debate_system=system_a_debate,
        arbiter_agent=arbiter,
    )

    for exp_name, sentence in experiments.items():
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"PROMPT: {sentence}")
        print(f"{'=' * 80}")

        print("\n[ RUNNING SYSTEM A: MULTI-AGENT DEBATE ]")
        try:
            result_a = system_a_debate.run(sentence)
            print(f"\n>>> SYSTEM A FINAL RESULT:\n{result_a}\n")
        except Exception as e:
            logger.error(f"System A failed during execution: {e}")

        print("\n[ RUNNING SYSTEM B: MAS + LOGIC VERIFIER ]")
        try:
            result_b = system_b_logic.run(sentence)
            print(f"\n>>> SYSTEM B FINAL RESULT:\n{result_b}\n")
        except Exception as e:
            logger.error(f"System B failed during execution: {e}")

        print("\n[ RUNNING SYSTEM C: COMBINED (MAS + VERIFIER + DEBATE) ]")
        try:
            result_c = system_c_combined.run(sentence)
            print(f"\n>>> SYSTEM C FINAL RESULT:\n{result_c}\n")
        except Exception as e:
            logger.error(f"System C failed during execution: {e}")


if __name__ == "__main__":
    main()
