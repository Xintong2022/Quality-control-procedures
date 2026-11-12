"""
cost_model.py — shared cost-model primitives for the TREK benchmark.

Imported by BOTH `scoring.py` (to bill an agent's plan for D3) and the query
generator (to compute C_min / C_floor when setting budgets). Keeping the unit
rules and the per-city night / car-day allocation in ONE place is what makes the
generator's budget labels correct-by-construction w.r.t. how the scorer grades —
the generator's C_min and the scorer's bill of the reference plan use the exact
same `city_maps(meta)`, so they cannot diverge.

Spec: DESIGN_trek_query_pipeline.md §A.

`_fold` is re-exported from `data_loader` (NOT re-implemented) so the folding used
to match a plan's hotel/car `city` against the query's `arrival_city` order is
byte-for-byte the same folding the KB city index uses. (Design hole H6.)
"""

from math import ceil

from data_loader import _fold  # re-export; must be the exact KB-index folding


def rooms_for(person_num):
    """2 guests per room (intentional KB assumption). rooms_count = ceil(person/2)."""
    return ceil(person_num / 2)


def stay_cities_in_order(meta):
    """Ordered destination cities the traveller stays in.

    Read from `meta.req_flight.arrival_city` (a list in canonical visiting order;
    tolerates a bare string for legacy rows). The departure city is NOT a stay city.
    """
    arr = (meta.req_flight or {}).get("arrival_city", [])
    if isinstance(arr, str):
        arr = [arr]
    return [c for c in (arr or []) if c]


def allocate(total, k):
    """Even split of `total` units across k cities; the remainder loads the EARLIER
    cities. `allocate(days-1, k)` -> nights per city; `allocate(days, k)` -> car-days.
    Sum is exactly `total` for any total >= 0, k >= 1.
    """
    if k <= 0:
        return []
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def city_maps(meta):
    """Deterministic, QUERY-derived per-city nights & car-days.

    Returns `(cities, idx, nights, cardays, base_n, base_d)`:
      cities   ordered stay cities
      idx      {folded_city_name: position}
      nights   per-city nights,   sum == max(1, days-1)
      cardays  per-city car-days, sum == max(1, days)
      base_n   fallback nights for a plan hotel whose city is off-itinerary/unlabelled
      base_d   fallback car-days for same

    Single-city (k=1): nights == [max(1, days-1)], cardays == [max(1, days)], and
    base_n == max(1, days-1), base_d == max(1, days). Because BOTH the matched-city
    value and the fallback equal the old scalar `nights = days-1` / `car_days = days`,
    the single-city bill is bit-identical to the pre-patch scorer regardless of whether
    the plan's hotel/car `city` string matches — only multi-city (k>1) changes.
    """
    cities = stay_cities_in_order(meta)
    k = max(1, len(cities))
    nights = allocate(max(1, meta.days - 1), k)
    cardays = allocate(max(1, meta.days), k)
    idx = {_fold(c): i for i, c in enumerate(cities)}
    base_n = max(1, (meta.days - 1) // k)
    base_d = max(1, meta.days // k)
    return cities, idx, nights, cardays, base_n, base_d
