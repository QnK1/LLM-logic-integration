import json
import math
import random
from pathlib import Path

import pandas as pd


def load_and_prep_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["dep_time", "arr_time", "distance"])

    def hhmm_to_mins(time_float):
        if pd.isna(time_float):
            return 0
        time_int = int(time_float)
        hours = time_int // 100
        mins = time_int % 100
        return hours * 60 + mins

    df["dep_mins"] = df["dep_time"].apply(hhmm_to_mins)
    df["arr_mins"] = df["arr_time"].apply(hhmm_to_mins)

    df["price"] = (50 + df["distance"] * 0.12).round(2)

    df["dep_str"] = df["dep_time"].apply(
        lambda x: f"{int(x) // 100:02d}:{int(x) % 100:02d}"
    )
    df["arr_str"] = df["arr_time"].apply(
        lambda x: f"{int(x) // 100:02d}:{int(x) % 100:02d}"
    )

    return df


def find_all_routes(df, origin, dest, min_layover_mins=45, max_layover_mins=300):
    routes = []

    directs = df[(df["origin"] == origin) & (df["dest"] == dest)]
    for _, flight in directs.iterrows():
        duration = flight["arr_mins"] - flight["dep_mins"]
        if duration < 0:
            duration += 1440

        routes.append(
            {
                "type": "direct",
                "stops": 0,
                "legs": [flight.to_dict()],
                "total_price": flight["price"],
                "total_time_mins": duration,
                "departure_time": flight["dep_str"],
                "arrival_time": flight["arr_str"],
            }
        )

    leg1_df = df[df["origin"] == origin].add_prefix("l1_")
    leg2_df = df[df["dest"] == dest].add_prefix("l2_")

    connections = pd.merge(leg1_df, leg2_df, left_on="l1_dest", right_on="l2_origin")

    if not connections.empty:
        connections["layover"] = connections["l2_dep_mins"] - connections["l1_arr_mins"]

        connections.loc[connections["layover"] < 0, "layover"] += 1440

        valid_conn = connections[
            (connections["layover"] >= min_layover_mins)
            & (connections["layover"] <= max_layover_mins)
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
                    "stops": 1,
                    "total_price": round(conn["l1_price"] + conn["l2_price"], 2),
                    "total_time_mins": total_time,
                    "departure_time": conn["l1_dep_str"],
                    "arrival_time": conn["l2_arr_str"],
                    "layover_airport": conn["l1_dest"],
                }
            )

    return routes


def generate_problems(df, num_problems=20):
    airports = df["origin"].unique().tolist() + df["dest"].unique().tolist()
    airports = list(set(airports))

    problems = []
    attempts = 0

    while len(problems) < num_problems and attempts < 1000:
        attempts += 1

        origin, dest = random.sample(airports, 2)

        all_routes = find_all_routes(df, origin, dest)
        if not all_routes:
            continue

        optimize_for = random.choice(["price", "time"])

        if optimize_for == "price":
            best_route = min(all_routes, key=lambda x: x["total_price"])
        else:
            best_route = min(all_routes, key=lambda x: x["total_time_mins"])

        max_price = math.ceil(best_route["total_price"] * 1.15)
        max_time = math.ceil(best_route["total_time_mins"] * 1.15)
        max_stops = best_route["stops"]

        statement = (
            f"I need to fly from {origin} to {dest}. "
            f"My budget is strictly ${max_price}. "
            f"The total travel time (including layovers) must not exceed {max_time} minutes. "
            f"I can tolerate a maximum of {max_stops} stops. Find a valid flight plan."
        )

        problem = {
            "problem_id": f"P{len(problems) + 1:03d}",
            "problem_statement": statement,
            "ground_truth": {
                "solution_exists": True,
                "constraints": {
                    "origin": origin,
                    "destination": dest,
                    "max_price_usd": max_price,
                    "max_time_mins": max_time,
                    "max_stops": max_stops,
                },
                "expected_optimal": {
                    "departure_time": best_route["departure_time"],
                    "arrival_time": best_route["arrival_time"],
                    "total_time_mins": best_route["total_time_mins"],
                    "total_price_usd": best_route["total_price"],
                    "stops": best_route["stops"],
                    "layover_airport": best_route.get("layover_airport", None),
                },
            },
        }

        problems.append(problem)

    return problems


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    CSV_PATH = PROJECT_ROOT / "data/flights.csv"
    OUTPUT_FILE = PROJECT_ROOT / "data/flight_planning_problems.json"

    print("Loading data...")
    df = load_and_prep_data(CSV_PATH)
    print(f"Loaded {len(df)} flights.")

    print("Generating problems via brute-force search...")
    generated_problems = generate_problems(df, num_problems=20)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(generated_problems, f, indent=2)

    print(
        f"Successfully generated {len(generated_problems)} problems and saved to {str(OUTPUT_FILE)}."
    )
