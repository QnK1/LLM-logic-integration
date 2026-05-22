import json
from typing import Any, List

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from llm_logic_integration.agents.logic_verifier_agent import LogicVerifierAgent


def create_train_search_tool(df: pd.DataFrame):
    """Factory function to inject the DataFrame into the LangChain tool."""

    @tool
    def search_train_connections(origin: str, destination: str, date: str) -> str:
        """
        Search the railway timetable for direct and 1-stop train connections.

        Args:
            origin: Departure station name (e.g., 'London Paddington').
            destination: Arrival station name (e.g., 'Liverpool Lime Street').
            date: The date of travel in YYYY-MM-DD format.

        Returns:
            A JSON-formatted string of the top 5 best routes (sorted by time and price),
            or an error message if no routes are found.
        """
        routes = []

        day_df = df[df["date"] == date]
        if day_df.empty:
            return json.dumps({"error": f"No trains operating on {date}."})

        directs = day_df[(day_df["origin"] == origin) & (day_df["dest"] == destination)]
        for _, train in directs.iterrows():
            duration = train["arr_mins"] - train["dep_mins"]
            if duration < 0:
                duration += 1440

            routes.append(
                {
                    "type": "direct",
                    "transfers": 0,
                    "departure_time": train["dep_str"],
                    "arrival_time": train["arr_str"],
                    "total_time_mins": duration,
                    "total_price_gbp": train["price"],
                    "path": f"{origin} -> {destination}",
                }
            )

        leg1_df = day_df[day_df["origin"] == origin].add_prefix("l1_")
        leg2_df = day_df[day_df["dest"] == destination].add_prefix("l2_")

        connections = pd.merge(
            leg1_df, leg2_df, left_on="l1_dest", right_on="l2_origin"
        )

        if not connections.empty:
            connections["layover"] = (
                connections["l2_dep_mins"] - connections["l1_arr_mins"]
            )
            connections.loc[connections["layover"] < 0, "layover"] += 1440

            valid_conn = connections[
                (connections["layover"] >= 10) & (connections["layover"] <= 120)
            ]

            for _, conn in valid_conn.iterrows():
                total_time = (
                    (conn["l1_arr_mins"] - conn["l1_dep_mins"])
                    + conn["layover"]
                    + (conn["l2_arr_mins"] - conn["l2_dep_mins"])
                )
                if total_time < 0:
                    total_time += 1440

                routes.append(
                    {
                        "type": "1-stop",
                        "transfers": 1,
                        "departure_time": conn["l1_dep_str"],
                        "arrival_time": conn["l2_arr_str"],
                        "transfer_station": conn["l1_dest"],
                        "layover_time_mins": int(conn["layover"]),
                        "total_time_mins": int(total_time),
                        "total_price_gbp": round(
                            conn["l1_price"] + conn["l2_price"], 2
                        ),
                        "path": f"{origin} -> {conn['l1_dest']} -> {destination}",
                    }
                )

        if not routes:
            return json.dumps(
                {
                    "error": f"No valid routes found from {origin} to {destination} on {date}."
                }
            )

        routes.sort(key=lambda x: (x["total_time_mins"], x["total_price_gbp"]))

        best_routes = routes[:5]

        return json.dumps(
            {
                "status": "success",
                "results_found": len(routes),
                "top_5_routes": best_routes,
            }
        )

    return search_train_connections


class TrainSegment(BaseModel):
    origin: str = Field(description="Departure station")
    destination: str = Field(description="Arrival station")
    date: str = Field(description="Date of travel (YYYY-MM-DD)")
    departure_time: str = Field(
        description="Departure time in HH:MM or HH:MM:SS format"
    )
    price: float = Field(description="Proposed price of this specific segment")


class VerifyFactsInput(BaseModel):
    segments: List[TrainSegment] = Field(
        description="List of train segments to verify against the database."
    )


def create_fact_verification_tool(df: pd.DataFrame):
    """Factory to inject the DataFrame into the fact checker."""

    @tool(args_schema=VerifyFactsInput)
    def verify_train_facts(segments: List[TrainSegment]) -> str:
        """
        Verify if the proposed train segments actually exist in the real database.
        Always run this to check for hallucinations before confirming a plan.
        """
        feedback = []
        all_valid = True

        for i, segment in enumerate(segments):
            # Extract only HH:MM to prevent matching errors between "17:00" and "17:00:00"
            safe_time = str(segment.departure_time)[:5]

            match = df[
                (df["origin"] == segment.origin)
                & (df["dest"] == segment.destination)
                & (df["date"] == segment.date)
                & (df["dep_str"].astype(str).str[:5] == safe_time)
            ]

            if match.empty:
                all_valid = False
                feedback.append(
                    f"Segment {i + 1} HALLUCINATED: No train exists from {segment.origin} to {segment.destination} "
                    f"on {segment.date} at {segment.departure_time}."
                )
                continue

            actual_price = match.iloc[0]["price"]
            if abs(actual_price - segment.price) > 0.1:
                all_valid = False
                feedback.append(
                    f"Segment {i + 1} PRICE MISMATCH: Train exists, but actual price is £{actual_price}, "
                    f"not £{segment.price}."
                )

        if all_valid:
            return json.dumps(
                {
                    "status": "SUCCESS",
                    "message": "All facts verified. The itinerary exists.",
                }
            )

        return json.dumps(
            {
                "status": "FAILURE",
                "message": "Found factual errors in the proposed plan.",
                "errors": feedback,
            }
        )

    return verify_train_facts


class VerifyConstraintsInput(BaseModel):
    proposed_price: float = Field(description="Total price of the proposed itinerary")
    max_budget: float = Field(description="Maximum allowed budget")
    proposed_time_mins: int = Field(
        description="Total travel time of the proposed itinerary in minutes"
    )
    max_time_mins: int = Field(description="Maximum allowed travel time in minutes")
    proposed_transfers: int = Field(description="Total number of transfers/stops")
    max_transfers: int = Field(description="Maximum allowed transfers")


@tool(args_schema=VerifyConstraintsInput)
def verify_plan_constraints(
    proposed_price: float,
    max_budget: float,
    proposed_time_mins: int,
    max_time_mins: int,
    proposed_transfers: int,
    max_transfers: int,
) -> str:
    """
    Verify if the proposed plan logically satisfies all numerical constraints (budget, time, transfers).
    """
    violations = []

    if proposed_price > max_budget:
        violations.append(
            f"BUDGET VIOLATION: Proposed price (£{proposed_price}) exceeds max budget (£{max_budget})."
        )

    if proposed_time_mins > max_time_mins:
        violations.append(
            f"TIME VIOLATION: Proposed time ({proposed_time_mins} mins) exceeds max time ({max_time_mins} mins)."
        )

    if proposed_transfers > max_transfers:
        violations.append(
            f"TRANSFER VIOLATION: Proposed transfers ({proposed_transfers}) exceeds max allowed ({max_transfers})."
        )

    if not violations:
        return json.dumps(
            {"status": "SUCCESS", "message": "All constraints satisfied."}
        )

    return json.dumps(
        {
            "status": "FAILURE",
            "message": "The plan violates the rules.",
            "violations": violations,
        }
    )


class LogicVerificationInput(BaseModel):
    original_sentence: str = Field(
        description="The original user prompt containing all planning requirements and constraints."
    )
    generator_answer: str | dict[str, Any] | list[Any] = Field(
        description="The full proposed itinerary and travel plan that needs logical proof validation."
    )


def create_logic_verification_tool(verifier_agent: LogicVerifierAgent):
    """
    Factory function to wrap the LogicVerifierAgent into a standard LangChain tool.
    Injects the agent instance into the tool's execution context.
    """

    @tool(args_schema=LogicVerificationInput)
    def verify_plan_logic(
        original_sentence: str, generator_answer: str | dict[str, Any] | list[Any]
    ) -> str:
        """
        Run a formal symbolic logic check (using an SMT Solver) on the proposed travel plan.
        Use this tool to prove that the current itinerary mathematically and logically
        satisfies all constraints from the original request.
        """
        if isinstance(generator_answer, (dict, list)):
            generator_answer_str = json.dumps(generator_answer)
        else:
            generator_answer_str = str(generator_answer)

        evaluation_result = verifier_agent.verify(
            original_sentence=original_sentence, generator_answer=generator_answer_str
        )

        return evaluation_result.model_dump_json()

    return verify_plan_logic
