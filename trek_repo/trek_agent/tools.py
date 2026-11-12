"""
Tool definitions (Bedrock toolConfig) + the dispatcher that executes them against the Travel API.

Billable tools (count toward D5 efficiency): search_flights/hotels/attractions/cars, submit_plan.
Free / non-billable tools (do NOT count toward D5): compute_travel_time (the exact model B3 scores
with) and write_note (the agent's notebook). See trek_agent/agent.py for the accounting.
"""
import json
import requests
from typing import Any, Optional

from .notebook import Notebook


AUX_TOOLS = {"compute_travel_time", "write_note"}     # free, never counted against efficiency
TERMINAL_TOOLS = {"submit_plan"}
SEARCH_TOOLS = {"search_flights", "search_hotels", "search_attractions", "search_cars"}


# Neutral tool specs (converted to Bedrock toolSpec below). Descriptions are terse; the system prompt
# carries the strategy.
TOOLS: list[dict] = [
    {
        "name": "search_flights",
        "description": "Search flights between two cities in the sandbox database.",
        "schema": {
            "type": "object",
            "properties": {
                "departure_city": {"type": "string", "description": "City to depart from."},
                "arrival_city": {"type": "string", "description": "Destination city."},
                "trip_type": {"type": "string", "enum": ["one_way", "round_trip"], "description": "one_way or round_trip."},
                "max_price": {"type": "number", "description": "Max price filter."},
                "sort_by": {"type": "string", "enum": ["price", "departure_time"]},
                "top_k": {"type": "integer", "description": "Number of results (default 20)."},
            },
            "required": ["departure_city", "arrival_city", "trip_type"],
        },
    },
    {
        "name": "search_hotels",
        "description": "Search hotels in a city. Pass ONE city, or comma-separated cities for a batch.",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City, or 'Paris,Rome' for several."},
                "name": {"type": "string", "description": "Look up a SPECIFIC hotel by name. Use this whenever the request names a hotel — it is the only reliable way to confirm whether that exact hotel exists, since a browse only returns top_k rows."},
                "max_price": {"type": "number", "description": "Max price per night."},
                "min_star": {"type": "integer", "description": "Minimum star rating (1-5)."},
                "amenity": {"type": "string", "description": "Amenity to look for (semantic)."},
                "sort_by": {"type": "string", "enum": ["price", "rating", "star"]},
                "top_k": {"type": "integer", "description": "Results per city (default 20)."},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_attractions",
        "description": "Search attractions in a city. Pass ONE city, or comma-separated for a batch.",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City, or 'Paris,Rome' for several."},
                "attraction_name": {"type": "string", "description": "Look up a SPECIFIC attraction by name. Use this whenever the request names an attraction — a browse only returns top_k rows, so absence from that list does NOT mean the attraction is missing."},
                "max_ticket_price": {"type": "number", "description": "Max ticket price."},
                "facility": {"type": "string", "description": "Facility to look for (semantic)."},
                "sort_by": {"type": "string", "enum": ["price", "rating", "duration_of_visit"]},
                "top_k": {"type": "integer", "description": "Results per city (default 20)."},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_cars",
        "description": "Search rental cars in a city. Pass ONE city, or comma-separated for a batch.",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City, or 'Paris,Rome' for several."},
                "min_capacity": {"type": "integer", "description": "Minimum passenger capacity."},
                "max_price_per_day": {"type": "number", "description": "Max price per day."},
                "car_type": {"type": "string", "description": "Car type."},
                "extra_service": {"type": "string", "description": "Extra service to look for (semantic)."},
                "top_k": {"type": "integer", "description": "Results per city (default 20)."},
            },
            "required": ["city"],
        },
    },
    {
        "name": "compute_travel_time",
        "description": (
            "FREE. Distance + the MINIMUM door-to-door minutes the feasibility check (B3) requires "
            "between two locations — the exact model the scorer uses. Verify consecutive same-day "
            "hops before submitting. Does NOT count against your tool budget. Identify each endpoint "
            "by *_latitude/*_longitude from a search result, or by *_name/*_city/*_type."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "from_name": {"type": "string"}, "from_city": {"type": "string"},
                "from_type": {"type": "string", "enum": ["attraction", "hotel", "flight"],
                              "description": "For 'flight' the name may be a flight number, an airport name, or an IATA code. For 'hotel'/'attraction' you must also pass the matching *_city."},
                "from_latitude": {"type": "number"}, "from_longitude": {"type": "number"},
                "to_name": {"type": "string"}, "to_city": {"type": "string"},
                "to_type": {"type": "string", "enum": ["attraction", "hotel", "flight"],
                            "description": "For 'flight' the name may be a flight number, an airport name, or an IATA code. For 'hotel'/'attraction' you must also pass the matching *_city."},
                "to_latitude": {"type": "number"}, "to_longitude": {"type": "number"},
            },
            "required": [],
        },
    },
    {
        "name": "write_note",
        "description": (
            "FREE. Jot a short note to your own planning notebook (e.g. a chosen flight+times, a "
            "day→city assignment, an opening-hours constraint). You will see all your notes again at "
            "final-planning time. Does NOT count against your tool budget."
        ),
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The note to remember."}},
            "required": ["text"],
        },
    },
    {
        "name": "submit_plan",
        "description": "Submit the final travel plan (or a refusal for a genuinely impossible task).",
        "schema": {
            "type": "object",
            "properties": {
                "is_feasible": {"type": "boolean", "description": "true = a real plan; false = refuse."},
                "refusal_reason": {"type": "string", "description": "If not feasible, which impossibility (budget / no route / entity does not exist)."},
                "plan": {
                    "type": "object",
                    "description": (
                        "Day-keyed object: day1, day2, … Each day has current_city ('A' or 'A to B'), "
                        "flights [{flight_number, departure_city, arrival_city, departure_time, "
                        "arrival_time, price}], attractions [{name, city, visit_start, visit_end}], "
                        "hotel {name, city, price_per_night, check_in}, car {car_type, capacity, "
                        "price_per_day, city}. Use ONLY entities returned by the search tools, with "
                        "their exact names and times. "
                        "EVERY time field — departure_time, arrival_time, visit_start, visit_end and "
                        "the hotel's check_in — must be a 24-hour clock time 'HH:MM' (e.g. '15:00'). "
                        "The day is already given by the day key, so do NOT write 'day1' or "
                        "'already checked in' in check_in."
                    ),
                },
            },
            "required": ["is_feasible", "plan"],
        },
    },
]

_TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def to_bedrock_toolconfig() -> dict:
    """Convert TOOLS to a Bedrock Converse `toolConfig`."""
    return {
        "tools": [
            {"toolSpec": {"name": t["name"], "description": t["description"],
                          "inputSchema": {"json": t["schema"]}}}
            for t in TOOLS
        ]
    }


class ToolDispatcher:
    """Executes a tool call against the Travel API / notebook and returns a JSON-able result."""

    def __init__(self, base_url: str, notebook: Notebook, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.notebook = notebook
        self.timeout = timeout

    # ---- public --------------------------------------------------------------------------------
    def execute(self, name: str, args: dict) -> Any:
        args = args or {}
        try:
            if name == "write_note":
                self.notebook.add_note(args.get("text", ""))
                return {"ok": True, "notes_total": len(self.notebook.notes)}
            if name == "compute_travel_time":
                return self._get("/travel_time", args)
            if name == "search_flights":
                return self._search_flights(args)
            if name in ("search_hotels", "search_attractions", "search_cars"):
                return self._search_city(name, args)
            return {"error": f"unknown tool: {name}"}
        except requests.exceptions.RequestException as e:
            # TRANSPORT failure = the Travel API is down/unreachable. Tagged distinctly from an
            # HTTP 4xx (which means the MODEL sent a bad request and is a legitimate result it
            # should learn from) so the driver can retry only the former.
            return {"error": f"api_error: {e}", "type": "api_down"}
        except Exception as e:  # never let a tool crash the whole agent turn
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- search impls --------------------------------------------------------------------------
    def _search_flights(self, args: dict) -> Any:
        params = {k: v for k, v in args.items() if v not in (None, "")}
        params.setdefault("trip_type", "one_way")
        result = self._get("/flights2", params)
        self.notebook.record_search("search_flights", result)
        return result

    def _search_city(self, name: str, args: dict) -> Any:
        endpoint = {"search_hotels": "/hotels2", "search_attractions": "/attractions2",
                    "search_cars": "/cars2"}[name]
        city = str(args.get("city", "")).strip()
        rest = {k: v for k, v in args.items() if k != "city" and v not in (None, "")}
        if "," in city:
            by_city = {}
            for c in [x.strip() for x in city.split(",") if x.strip()]:
                r = self._get(endpoint, {"city": c, **rest})
                by_city[c] = r
                self.notebook.record_search(name, r)
            key = {"search_hotels": "hotels_by_city", "search_attractions": "attractions_by_city",
                   "search_cars": "cars_by_city"}[name]
            return {key: by_city}
        result = self._get(endpoint, {"city": city, **rest})
        self.notebook.record_search(name, result)
        return result

    # ---- http ----------------------------------------------------------------------------------
    def _get(self, endpoint: str, params: dict) -> Any:
        resp = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=self.timeout)
        # surface HTTP errors as structured results instead of raising, so the model can react
        if resp.status_code >= 400:
            body = _safe_json(resp)
            # 5xx is the SERVER failing; 4xx is the model having sent a bad request (e.g. omitting
            # the required `city`), which is a normal result the agent should see and correct.
            kind = "api_down" if resp.status_code >= 500 else "api_error"
            return {"error": f"HTTP {resp.status_code}", "type": kind, "detail": body}
        return _safe_json(resp)


def _safe_json(resp) -> Any:
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        return {"error": "non-JSON response", "text": resp.text[:500]}
