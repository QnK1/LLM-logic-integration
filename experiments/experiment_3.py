import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field

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

    systems = {
        "MULTI-AGENT DEBATE": system_a_debate,
        "MAS + LOGIC VERIFIER": system_b_logic,
        "COMBINED (MAS + VERIFIER + DEBATE)": system_c_combined,
    }

    stats = {
        sys_name: {
            "runs": 0,
            "successes": 0,
            "failures": 0,
            "errors": 0,
        }
        for sys_name in systems.keys()
    }

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

    for exp_name, exp_data in experiments.items():
        sentence = exp_data["prompt"]
        schema = exp_data["schema"]
        truth = exp_data["truth"]

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: {exp_name}")
        print(f"PROMPT: {sentence}")
        print(f"{'=' * 80}")

        for sys_name, system in systems.items():
            print(f"\n[ RUNNING SYSTEM: {sys_name} ]")

            for i in range(ITERATIONS_PER_EXPERIMENT):
                stats[sys_name]["runs"] += 1
                try:
                    result = system.run(sentence, output_schema=schema)  # ty:ignore[invalid-argument-type]
                    result_dump = result.model_dump()
                    print(f"  Iteration {i + 1} Result: {result_dump}")

                    if check_result(result_dump, truth):  # ty:ignore[invalid-argument-type]
                        stats[sys_name]["successes"] += 1
                    else:
                        stats[sys_name]["failures"] += 1

                except Exception as e:
                    logger.error(f"  Iteration {i + 1} failed: {e}")
                    stats[sys_name]["errors"] += 1

    print(f"\n{'=' * 80}")
    print("EXPERIMENT STATS")
    print(f"{'=' * 80}")

    for sys_name, sys_stats in stats.items():
        runs = sys_stats["runs"]
        successes = sys_stats["successes"]
        failures = sys_stats["failures"]
        errors = sys_stats["errors"]

        acc = (successes / runs) * 100 if runs > 0 else 0

        print(f"\nSYSTEM: {sys_name}")
        print(f"  Total Runs: {runs}")
        print(f"  Successes:  {successes} ({acc:.1f}%)")
        print(f"  Failures:   {failures}")
        print(f"  Errors:     {errors}")


if __name__ == "__main__":
    main()
