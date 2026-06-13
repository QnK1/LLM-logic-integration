import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

from llm_logic_integration.agents.arbiter_agent import ArbiterAgent
from llm_logic_integration.agents.critic_agent import CriticAgent
from llm_logic_integration.agents.generator_agent import GeneratorAgent
from llm_logic_integration.multi_agent_systems.mas_no_logic import MASNoLogic

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

PROVIDER = "ollama"
MODEL_NAME = "qwen2.5:7b-instruct"

ITERATIONS_TO_TEST = [1, 2, 3, 4]
ITERATIONS_PER_EXPERIMENT = 5


def get_api_key(provider: str) -> str | None:
    provider = provider.lower()
    if provider == "gemini":
        return str(os.getenv("GEMINI_API_KEY"))
    elif provider == "openai":
        return str(os.getenv("OPENAI_API_KEY"))
    return None


class CounterfactualOutput(BaseModel):
    floats: bool = Field(description="True if the penguin floats, False if it sinks.")


class MultiHopOutput(BaseModel):
    is_epsilon_critical: bool = Field(description="True if Epsilon is critical.")
    is_zeta_functioning: bool = Field(description="True if Zeta is functioning.")


class ConstraintOutput(BaseModel):
    is_order_possible: bool = Field(
        description="True if the order B, A, D, C is possible."
    )


class NegationOutput(BaseModel):
    is_secure: bool = Field(description="True if System X is secure.")


class ParityOutput(BaseModel):
    other_door_color: str = Field(
        description="The exact color of the other door, or 'None'/'Unsatisfiable' if it is a paradox."
    )


def check_result(result_dict: dict, truth_dict: dict) -> bool:
    for k, v in truth_dict.items():
        res_val = result_dict.get(k)
        if isinstance(res_val, str) and isinstance(v, str):
            if v.lower() not in res_val.lower():
                return False
        else:
            if res_val != v:
                return False
    return True


def main():
    load_dotenv()
    api_key = str(get_api_key(PROVIDER))

    experiments = {
        "1. Counterfactuals": {
            "prompt": (
                "All birds are made of liquid metal. Anything made of liquid metal sinks in water. "
                "Penguins are a type of bird. A standard wooden boat floats on water. "
                "Does a penguin float or sink in water?"
            ),
            "schema": CounterfactualOutput,
            "truth": {"floats": False},
        },
        "2. Deep Multi-Hop Reasoning": {
            "prompt": (
                "If Alpha is active, then Beta is dormant. If Gamma is unstable, then Delta is active. "
                "If Beta is dormant, then Epsilon is critical. Zeta is always functioning unless Delta is active. "
                "Alpha is active. Gamma is stable. Is Epsilon critical, and is Zeta functioning?"
            ),
            "schema": MultiHopOutput,
            "truth": {"is_epsilon_critical": True, "is_zeta_functioning": True},
        },
        "3. Constraint Satisfaction": {
            "prompt": (
                "We have four tasks: A, B, C, and D. Task A must be completed before Task C. "
                "Task B cannot be the first or the last task. Task D must be done immediately after Task A. "
                "Can the order be B, A, D, C?"
            ),
            "schema": ConstraintOutput,
            "truth": {"is_order_possible": False},
        },
        "4. Structural Negation": {
            "prompt": (
                "If a software system is secure, it must be encrypted. "
                "If a system is encrypted, it uses more processing power. "
                "System X does not use more processing power. Is System X secure?"
            ),
            "schema": NegationOutput,
            "truth": {"is_secure": False},
        },
        "5. Parity and Counting": {
            "prompt": (
                "There are five doors: Red, Blue, Green, Yellow, and Black. Exactly two doors lead to the prize. "
                "If the Red door leads to the prize, the Green door does not. "
                "The Blue door and the Yellow door have the exact same outcome. "
                "The Black door does not lead to the prize. The Green door leads to the prize. "
                "Which other door leads to the prize?"
            ),
            "schema": ParityOutput,
            "truth": {"other_door_color": "none"},
        },
    }

    logger.info(f"Initializing agents using {PROVIDER.upper()} model: {MODEL_NAME}")

    generator = GeneratorAgent(
        provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key
    )
    critic = CriticAgent(provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key)
    arbiter = ArbiterAgent(provider=PROVIDER, model_name=MODEL_NAME, api_key=api_key)

    mas_no_logic = MASNoLogic(
        max_iterations=1,
        generator=generator,
        critic=critic,
        arbiter=arbiter,
        force_max_iter=True,
    )

    systems = {
        "MAS (NO LOGIC)": mas_no_logic,
    }

    stats = {
        sys_name: {
            iter_count: {"runs": 0, "successes": 0, "failures": 0, "errors": 0}
            for iter_count in ITERATIONS_TO_TEST
        }
        for sys_name in systems.keys()
    }

    for exp_name, exp_data in experiments.items():
        sentence = exp_data["prompt"]
        schema = exp_data["schema"]
        truth = exp_data["truth"]

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"PROMPT: {sentence}")
        print(f"{'=' * 80}")

        for sys_name, system in systems.items():
            print(f"\n[ SYSTEM: {sys_name} ]")

            for max_iter in ITERATIONS_TO_TEST:
                system.max_iterations = max_iter
                print(f"  --- Testing Plateau at exactly {max_iter} iteration(s) ---")

                for i in range(ITERATIONS_PER_EXPERIMENT):
                    stats[sys_name][max_iter]["runs"] += 1
                    try:
                        result = system.run(sentence, output_schema=schema)  # ty:ignore[invalid-argument-type]
                        result_dump = result.model_dump()
                        print(f"    Run {i + 1} Result: {result_dump}")

                        if check_result(result_dump, truth):  # ty:ignore[invalid-argument-type]
                            stats[sys_name][max_iter]["successes"] += 1
                        else:
                            stats[sys_name][max_iter]["failures"] += 1

                    except Exception as e:
                        logger.error(f"    Run {i + 1} failed: {e}")
                        stats[sys_name][max_iter]["errors"] += 1

    print(f"\n{'=' * 80}")
    print("EXPERIMENT 4 STATS: PLATEAU ANALYSIS")
    print(f"{'=' * 80}")

    for sys_name, sys_stats in stats.items():
        print(f"\nSYSTEM: {sys_name}")
        print(
            f"{'Iterations':<15} | {'Accuracy (%)':<15} | {'Successes':<10} | {'Runs':<10}"
        )
        print("-" * 60)

        for iter_count in ITERATIONS_TO_TEST:
            runs = sys_stats[iter_count]["runs"]
            successes = sys_stats[iter_count]["successes"]

            acc = (successes / runs) * 100 if runs > 0 else 0

            print(f"{iter_count:<15} | {acc:<15.1f} | {successes:<10} | {runs:<10}")


if __name__ == "__main__":
    main()
