"""
The agent's working memory ("笔记本").

Two layers:
  1. A LOSSLESS entity ledger — every flight/hotel/attraction/car the search tools return is recorded
     with ALL its fields (flight departure/arrival times, attraction open_hours/duration/lat-lon, hotel
     check-in, car price/capacity). The monolithic engine threw these away and forced the model to
     fabricate times at submit_plan; here nothing collected is ever dropped. The ledger is also in the
     model's conversation history verbatim — this is the structured, deduped view used to build the
     final-planning brief.
  2. Model-authored NOTES — free text the agent appends via the write_note tool as it decides things
     ("chosen outbound BA123 dep 08:00", "Day2 = Rome, museum opens 09:00"). This is the notebook the
     model actually writes in and re-reads before composing the final plan.

Nothing here calls the network; it's pure in-memory state owned by one TrekAgent run.
"""
from dataclasses import dataclass, field
from typing import Any


def _identity(kind: str, item: dict) -> tuple:
    """A dedup key so re-searching the same city doesn't bloat the ledger, without dropping fields."""
    g = lambda *ks: next((str(item[k]) for k in ks if k in item and item[k] not in (None, "")), "")
    if kind == "flight":
        return ("flight", g("flight_number", "flightNumber").lower())
    if kind == "hotel":
        return ("hotel", g("name", "hotel_name").lower(), g("city", "city_name").lower())
    if kind == "car":
        return ("car", g("car_type", "type").lower(), g("city", "city_name").lower(), g("price_per_day"))
    if kind == "attraction":
        return ("attraction", g("attraction_name", "name").lower(), g("city", "city_name").lower())
    return (kind, repr(sorted(item.items())))


@dataclass
class Notebook:
    # kind -> identity -> full record (last write wins, but records are additive)
    entities: dict = field(default_factory=lambda: {"flight": {}, "hotel": {}, "attraction": {}, "car": {}})
    notes: list = field(default_factory=list)

    # ---- ingest --------------------------------------------------------------------------------
    def record_search(self, tool_name: str, result: Any) -> int:
        """Fold a search tool's result into the ledger. Returns how many new entities were added."""
        kind = {
            "search_flights": "flight",
            "search_hotels": "hotel",
            "search_attractions": "attraction",
            "search_cars": "car",
        }.get(tool_name)
        if kind is None:
            return 0
        added = 0
        for item in _iter_entities(result):
            if not isinstance(item, dict):
                continue
            key = _identity(kind, item)
            if key not in self.entities[kind]:
                added += 1
            # keep the fullest record we've seen for this identity
            merged = dict(self.entities[kind].get(key, {}))
            merged.update({k: v for k, v in item.items() if v not in (None, "")})
            self.entities[kind][key] = merged
        return added

    def add_note(self, text: str) -> None:
        if text and str(text).strip():
            self.notes.append(str(text).strip())

    # ---- counts / rendering --------------------------------------------------------------------
    def counts(self) -> dict:
        return {k: len(v) for k, v in self.entities.items()}

    def render_for_planning(self, max_per_kind: int = 40) -> str:
        """A compact-but-lossless brief for the final-planning turn: the agent's own notes first,
        then the full collected inventory with the scorer-relevant fields spelled out."""
        out = []
        if self.notes:
            out.append("## Your notes")
            out.extend(f"- {n}" for n in self.notes)
        out.append("\n## Everything you collected (use ONLY these exact entities)")
        for kind, label in (("flight", "Flights"), ("hotel", "Hotels"),
                            ("attraction", "Attractions"), ("car", "Cars")):
            recs = list(self.entities[kind].values())
            if not recs:
                continue
            out.append(f"\n### {label} ({len(recs)})")
            for r in recs[:max_per_kind]:
                out.append("- " + _fmt(kind, r))
            if len(recs) > max_per_kind:
                out.append(f"  … and {len(recs) - max_per_kind} more (see earlier tool results).")
        return "\n".join(out)

    def to_dict(self) -> dict:
        """Serializable snapshot for the output record (audit trail of what the agent had)."""
        return {
            "counts": self.counts(),
            "notes": list(self.notes),
            "entities": {k: list(v.values()) for k, v in self.entities.items()},
        }


def _iter_entities(result: Any):
    """Yield entity dicts from any search-result shape the Flask API returns."""
    if isinstance(result, list):
        yield from result
        return
    if not isinstance(result, dict):
        return
    # multi-city containers: {"hotels_by_city": {city: [...]}}, likewise attractions/cars/flights
    for key in ("hotels_by_city", "attractions_by_city", "cars_by_city", "flights_by_city"):
        if key in result and isinstance(result[key], dict):
            for _city, lst in result[key].items():
                if isinstance(lst, list):
                    yield from lst
            return
    # single-city containers
    for key in ("flights", "depart_flights", "return_flights", "hotels", "attractions", "cars", "data"):
        if key in result and isinstance(result[key], list):
            yield from result[key]
    # a bare entity dict
    if not any(k in result for k in ("flights", "hotels", "attractions", "cars", "data",
                                     "hotels_by_city", "attractions_by_city", "cars_by_city")):
        if any(k in result for k in ("flight_number", "name", "attraction_name", "car_type")):
            yield result


def _fmt(kind: str, r: dict) -> str:
    """One ledger line. Carries the fields the plan is GRADED on, not a readable subset.

    The brief tells the model to build its plan "using ONLY the exact entities above", so anything
    dropped here is invisible at the moment of choosing. Three field groups were being stripped from
    100% of records and each is scored:
      - amenities / facilities / extra_services -> D1 matches the traveller persona on exactly these,
        so a persona-targeted choice was being asked for from a list with the persona evidence
        deleted;
      - hotel latitude/longitude -> B3 needs them for every hotel hop, and without them here the
        model had to re-search or guess;
      - flight airport names -> the natural argument to compute_travel_time for an airport hop.
    """
    g = lambda *ks: next((r[k] for k in ks if k in r and r[k] not in (None, "")), None)

    def _l(*ks, limit=6):
        """Render a list-ish field compactly, or '' if absent."""
        v = g(*ks)
        if v in (None, "", [], "[]"):
            return ""
        if isinstance(v, str):
            import ast
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return f", {v}"
        if isinstance(v, (list, tuple)):
            items = [str(x) for x in v if str(x).strip()][:limit]
            return (", " + "; ".join(items)) if items else ""
        return f", {v}"

    if kind == "flight":
        return (f"{g('flight_number','flightNumber')}: {g('departure_city')}→{g('arrival_city')} "
                f"dep {g('departure_time')} arr {g('arrival_time')} ${g('price')} "
                f"[{g('departure_airport_name')} → {g('arrival_airport_name')}]")
    if kind == "hotel":
        return (f"{g('name','hotel_name')} ({g('city','city_name')}): ${g('price')}/night, "
                f"{g('star')}*, rating {g('rating')}, "
                f"lat/lon {g('latitude')},{g('longitude')}{_l('amenities','amenity')}")
    if kind == "attraction":
        return (f"{g('attraction_name','name')} ({g('city','city_name')}): ${g('ticket_price')}, "
                f"open {g('open_hours')}, needs {g('duration_of_visit')}, "
                f"lat/lon {g('latitude')},{g('longitude')}{_l('facilities','facility')}")
    if kind == "car":
        return (f"{g('car_type','type')} ({g('city','city_name')}): ${g('price_per_day')}/day, "
                f"{g('capacity')} seats{_l('extra_services','extra_service')}")
    return str(r)
