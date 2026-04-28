import os

from dotenv import load_dotenv
from loguru import logger

from llm_logic_integration.agents.arbiter_agent import ArbiterAgent
from llm_logic_integration.agents.critic_agent import CriticAgent
from llm_logic_integration.agents.generator_agent import GeneratorAgent
from llm_logic_integration.agents.logic_verifier_agent import LogicVerifierAgent
from llm_logic_integration.multi_agent_systems.mas_no_logic import MASNoLogic
from llm_logic_integration.multi_agent_systems.mas_with_logic import MASWithLogic
from llm_logic_integration.solvers.nltk_solver import NLTKSolver

# --- GLOBAL CONFIGURATION ---
PROVIDER = "ollama"  # Options: "gemini", "openai", "ollama"
MODEL_NAME = (
    "llama3"  # Examples: "gemini-2.5-flash-lite", "gpt-4o-mini", "llama3", "mistral"
)


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

    solver = NLTKSolver()
    logic_verifier = LogicVerifierAgent(
        provider=PROVIDER, model_name=MODEL_NAME, solver=solver, api_key=api_key
    )

    mas_no_logic = MASNoLogic(
        max_iterations=3, generator=generator, critic=critic, arbiter=arbiter
    )
    mas_with_logic = MASWithLogic(
        max_iterations=3,
        generator=generator,
        critic=critic,
        logic_verifier=logic_verifier,
        arbiter=arbiter,
    )

    for exp_name, sentence in experiments.items():
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"PROMPT: {sentence}")
        print(f"{'=' * 80}")

        print("\n[ RUNNING SYSTEM A: MAS (NO LOGIC) ]")
        try:
            result_a = mas_no_logic.run(sentence)
            print(f"\n>>> SYSTEM A FINAL RESULT:\n{result_a}\n")
        except Exception as e:
            logger.error(f"System A failed during execution: {e}")

        print("\n[ RUNNING SYSTEM B: MAS + LOGIC VERIFIER ]")
        try:
            result_b = mas_with_logic.run(sentence)
            print(f"\n>>> SYSTEM B FINAL RESULT:\n{result_b}\n")
        except Exception as e:
            logger.error(f"System B failed during execution: {e}")


if __name__ == "__main__":
    main()
