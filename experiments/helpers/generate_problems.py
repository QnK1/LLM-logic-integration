import json
import math
import random
from pathlib import Path

import pandas as pd


def load_and_prep_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    essential_cols = [
        "Departure Station",
        "Arrival Destination",
        "Departure Time",
        "Arrival Time",
        "Date of Journey",
        "Price",
    ]
    df = df.dropna(subset=essential_cols)

    df["Price"] = pd.to_numeric(
        df["Price"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
    )
    df = df.dropna(subset=["Price"])

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

    return df


def find_all_routes(df, origin, dest, min_layover_mins=10, max_layover_mins=120):
    routes = []

    directs = df[(df["origin"] == origin) & (df["dest"] == dest)]
    for _, train in directs.iterrows():
        duration = train["arr_mins"] - train["dep_mins"]
        if duration < 0:
            duration += 1440

        routes.append(
            {
                "type": "direct",
                "transfers": 0,
                "legs": [train.to_dict()],
                "total_price": train["price"],
                "total_time_mins": duration,
                "departure_time": train["dep_str"],
                "arrival_time": train["arr_str"],
                "date": train["date"],
            }
        )

    leg1_df = df[df["origin"] == origin].add_prefix("l1_")
    leg2_df = df[df["dest"] == dest].add_prefix("l2_")

    connections = pd.merge(
        leg1_df,
        leg2_df,
        left_on=["l1_dest", "l1_date"],
        right_on=["l2_origin", "l2_date"],
    )

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
                    "transfers": 1,
                    "total_price": round(conn["l1_price"] + conn["l2_price"], 2),
                    "total_time_mins": total_time,
                    "departure_time": conn["l1_dep_str"],
                    "arrival_time": conn["l2_arr_str"],
                    "transfer_station": conn["l1_dest"],
                    "date": conn["l1_date"],
                }
            )

    return routes


def generate_problems(df, num_problems=20):
    stations = df["origin"].unique().tolist() + df["dest"].unique().tolist()
    stations = list(set(stations))

    problems = []
    attempts = 0

    while len(problems) < num_problems and attempts < 1000:
        attempts += 1

        origin, dest = random.sample(stations, 2)

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
        max_transfers = best_route["transfers"]
        travel_date = best_route["date"]

        statement = (
            f"I need to travel by train from {origin} to {dest} on {travel_date}. "
            f"My budget is strictly £{max_price}. "
            f"The total travel time (including transfers) must not exceed {max_time} minutes. "
            f"I can tolerate a maximum of {max_transfers} transfers. Find a valid itinerary."
        )

        problem = {
            "problem_id": f"P{len(problems) + 1:03d}",
            "problem_statement": statement,
            "ground_truth": {
                "solution_exists": True,
                "constraints": {
                    "origin": origin,
                    "destination": dest,
                    "date": travel_date,
                    "max_price_gbp": max_price,
                    "max_time_mins": max_time,
                    "max_transfers": max_transfers,
                },
                "expected_optimal": {
                    "departure_time": best_route["departure_time"],
                    "arrival_time": best_route["arrival_time"],
                    "total_time_mins": best_route["total_time_mins"],
                    "total_price_gbp": best_route["total_price"],
                    "transfers": best_route["transfers"],
                    "transfer_station": best_route.get("transfer_station", None),
                },
            },
        }

        problems.append(problem)

    return problems


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    CSV_PATH = PROJECT_ROOT / "data/railway.csv"
    OUTPUT_FILE = PROJECT_ROOT / "data/railway_planning_problems.json"

    print("Loading railway data...")
    try:
        df = load_and_prep_data(CSV_PATH)
        print(f"Loaded {len(df)} unique scheduled trains from transaction logs.")

        print("Generating problems via brute-force search...")
        generated_problems = generate_problems(df, num_problems=20)

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(generated_problems, f, indent=2)

        print(
            f"Successfully generated {len(generated_problems)} problems and saved to {str(OUTPUT_FILE)}."
        )
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {CSV_PATH}. Please verify the path.")
