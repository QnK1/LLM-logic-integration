from pydantic import BaseModel, Field


class TravelPlan(BaseModel):
    journey_date: str = Field(
        description="Exact date of the journey in YYYY-MM-DD format."
    )
    departure_time: str = Field(
        description="Exact departure time in HH:MM or HH:MM:SS format ONLY (e.g., '17:00'). DO NOT include the date."
    )
    arrival_time: str = Field(
        description="Exact arrival time in HH:MM or HH:MM:SS format ONLY (e.g., '20:15'). DO NOT include the date."
    )
    total_time: int = Field(
        description="Total travel time in minutes (rounded if necessary)."
    )
    total_price_gbp: float = Field(description="Total travel price in GBP.")
    transfer_station: str | None = Field(
        description="Transfer station name, or exactly null/None if the train is direct."
    )
