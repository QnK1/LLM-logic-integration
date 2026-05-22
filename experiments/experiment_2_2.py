import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from llm_logic_integration.agents.logic_verifier_agent import LogicVerifierAgent
from llm_logic_integration.agents.travel_planning.planner_agent import (
    TravelPlannerAgent,
)
from llm_logic_integration.agents.travel_planning.tools import (
    create_fact_verification_tool,
    create_logic_verification_tool,
    create_train_search_tool,
    verify_plan_constraints,
)
from llm_logic_integration.solvers.z3_solver import Z3Solver

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

PROVIDER = "ollama"
MODEL_NAME = "qwen2.5:7b-instruct"


def get_api_key(provider: str) -> str | None:
    provider = provider.lower()
    if provider == "gemini":
        return str(os.getenv("GEMINI_API_KEY"))
    elif provider == "openai":
        return str(os.getenv("OPENAI_API_KEY"))
    return None


def main() -> None:
    load_dotenv()
    api_key = str(get_api_key(PROVIDER))

    df_path = Path(__file__).parent.parent / "data/railway.csv"
    df = pd.read_csv(df_path)

    def time_str_to_mins(time_str):
        if pd.isna(time_str):
            return 0
        parts = str(time_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])

    df["dep_mins"] = df["Departure Time"].apply(time_str_to_mins)
    df["arr_mins"] = df["Arrival Time"].apply(time_str_to_mins)
    df = df.rename(
        columns={
            "Departure Station": "origin",
            "Arrival Destination": "dest",
            "Departure Time": "dep_str",
            "Arrival Time": "arr_str",
            "Date of Journey": "date",
            "Price": "price",
        }
    )
    df = df.drop_duplicates(subset=["date", "origin", "dest", "dep_str", "arr_str"])

    solver = Z3Solver()
    logic_verifier_agent = LogicVerifierAgent(
        PROVIDER, MODEL_NAME, solver, api_key, max_retries=5
    )

    search_tool = create_train_search_tool(df)
    fact_verification_tool = create_fact_verification_tool(df)
    constraint_tool = verify_plan_constraints
    logic_tool = create_logic_verification_tool(logic_verifier_agent)

    base_agent = TravelPlannerAgent(
        PROVIDER, MODEL_NAME, tools=[search_tool], api_key=api_key
    )
    fact_agent = TravelPlannerAgent(
        PROVIDER,
        MODEL_NAME,
        tools=[search_tool, fact_verification_tool],
        api_key=api_key,
    )
    rule_agent = TravelPlannerAgent(
        PROVIDER, MODEL_NAME, tools=[search_tool, constraint_tool], api_key=api_key
    )
    logic_agent = TravelPlannerAgent(
        PROVIDER, MODEL_NAME, tools=[search_tool, logic_tool], api_key=api_key
    )

    agents = {
        "Base": base_agent,
        "Knowledge Verification": fact_agent,
        "Rule Verification": rule_agent,
        "Logic Verification": logic_agent,
    }

    problems_path = Path(__file__).parent.parent / "data/railway_planning_problems.json"
    with open(problems_path, "r", encoding="utf-8") as f:
        problems: list[dict] = json.load(f)

    for problem in problems:
        print(f"\n{'=' * 80}")
        print(f"PROBLEM: {problem['problem_id']}")
        print(f"PROMPT: {problem['problem_statement']}")
        print(f"{'=' * 80}")

        constraints = problem["ground_truth"]["constraints"]
        origin = constraints["origin"]
        destination = constraints["destination"]

        for name, agent in agents.items():
            print(f"\n>>> Running Agent: [ {name} ]")
            try:
                plan = agent.generate_plan(problem["problem_statement"])

                print("  [+] Generated Plan:")

                if hasattr(plan, "model_dump_json"):
                    plan_json = plan.model_dump_json(indent=4)
                else:
                    plan_json = plan.json(indent=4)

                for line in plan_json.split("\n"):
                    print(f"      {line}")
                print()

                transfer_st = plan.transfer_station
                has_transfer = bool(
                    transfer_st
                    and str(transfer_st).strip().lower()
                    not in ["", "none", "null", "n/a"]
                )

                actual_transfers = 1 if has_transfer else 0

                constraint_check_raw = constraint_tool.invoke(
                    {
                        "proposed_price": float(plan.total_price_gbp),
                        "max_budget": float(constraints["max_price_gbp"]),
                        "proposed_time_mins": int(plan.total_time),
                        "max_time_mins": int(constraints["max_time_mins"]),
                        "proposed_transfers": actual_transfers,
                        "max_transfers": int(constraints["max_transfers"]),
                    }
                )
                constraint_check = json.loads(constraint_check_raw)

                dep_time_str = str(plan.departure_time).strip()
                arr_time_str = str(plan.arrival_time).strip()
                generated_date = str(plan.journey_date).strip()

                tool_segments = []
                if not has_transfer:
                    tool_segments.append(
                        {
                            "origin": origin,
                            "destination": destination,
                            "date": generated_date,
                            "departure_time": dep_time_str,
                            "price": float(plan.total_price_gbp),
                        }
                    )
                else:
                    safe_dep_time = dep_time_str[:5]
                    safe_arr_time = arr_time_str[:5]

                    leg1_df = df[
                        (df["origin"] == origin)
                        & (df["dest"] == transfer_st)
                        & (df["date"] == generated_date)
                        & (df["dep_str"].astype(str).str[:5] == safe_dep_time)
                    ]

                    if leg1_df.empty:
                        tool_segments.append(
                            {
                                "origin": origin,
                                "destination": transfer_st,
                                "date": generated_date,
                                "departure_time": dep_time_str,
                                "price": float(plan.total_price_gbp),
                            }
                        )
                    else:
                        leg1 = leg1_df.iloc[0]
                        l1_arr_mins = leg1["arr_mins"]
                        l1_price = float(leg1["price"])

                        leg2_df = df[
                            (df["origin"] == transfer_st)
                            & (df["dest"] == destination)
                            & (df["date"] == generated_date)
                        ]

                        valid_leg2 = None
                        for _, l2_row in leg2_df.iterrows():
                            layover = l2_row["dep_mins"] - l1_arr_mins
                            if layover < 0:
                                layover += 1440

                            if (
                                10 <= layover <= 120
                                and str(l2_row["arr_str"])[:5] == safe_arr_time
                            ):
                                valid_leg2 = l2_row
                                break

                        if valid_leg2 is None:
                            tool_segments.append(
                                {
                                    "origin": origin,
                                    "destination": transfer_st,
                                    "date": generated_date,
                                    "departure_time": dep_time_str,
                                    "price": l1_price,
                                }
                            )
                            tool_segments.append(
                                {
                                    "origin": transfer_st,
                                    "destination": destination,
                                    "date": generated_date,
                                    "departure_time": "00:00",
                                    "price": round(
                                        float(plan.total_price_gbp) - l1_price, 2
                                    ),
                                }
                            )
                        else:
                            allocated_l2_price = round(
                                float(plan.total_price_gbp) - l1_price, 2
                            )

                            tool_segments.append(
                                {
                                    "origin": origin,
                                    "destination": transfer_st,
                                    "date": generated_date,
                                    "departure_time": dep_time_str,
                                    "price": l1_price,
                                }
                            )
                            tool_segments.append(
                                {
                                    "origin": transfer_st,
                                    "destination": destination,
                                    "date": generated_date,
                                    "departure_time": str(valid_leg2["dep_str"]),
                                    "price": allocated_l2_price,
                                }
                            )

                fact_check_raw = fact_verification_tool.invoke(
                    {"segments": tool_segments}
                )
                fact_check = json.loads(fact_check_raw)

                print(
                    f"  [-] Constraint Verification Status: {constraint_check['status']}"
                )
                if constraint_check["status"] == "FAILURE":
                    for violation in constraint_check["violations"]:
                        print(f"      * {violation}")

                print(f"  [-] Fact Verification Status: {fact_check['status']}")
                if fact_check["status"] == "FAILURE":
                    for error in fact_check["errors"]:
                        print(f"      * {error}")

            except Exception as e:
                logger.error(
                    f"Agent {name} failed to process problem {problem['problem_id']}: {e}"
                )


if __name__ == "__main__":
    main()
