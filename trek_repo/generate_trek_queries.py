"""
generate_trek_queries.py — TREK benchmark query generator (v2-grounded).

Produces the 800-task query set with ground-truth labels correct BY CONSTRUCTION:
budgets are set from the real cheapest cost of a v2-grounded reference plan, using
the SAME per-city cost model (`cost_model.city_maps`) the scorer bills with, and
every row is self-verified against the authoritative scorer before it ships.

Spec: DESIGN_trek_query_pipeline.md.

This file is built in sections:
  §1 KB index + pricing core  (route graph, cheapest_*, C_min_full/C_floor)  <-- this commit
  §2 sampling pipeline (5 stages) + self-verify + CSV output                 <-- next

Deterministic: seeded numpy RNG; reads only api/data/v2. Never hand-edit outputs.
"""

import os
import ast
import csv
import glob
import json
import math
import unicodedata
import numpy as np
import pandas as pd

from cost_model import city_maps, _fold, rooms_for, stay_cities_in_order

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(CUR_DIR, "api", "data", "v2")
FLIGHTS_CSV = os.path.join(V2, "flight_data", "flights.csv")

CAR_TYPES = ["Compact", "Convertible", "EV", "Economy", "Luxury", "Minivan", "SUV", "Sedan", "Wagon"]

# --------------------------------------------------------------------------------------
# §1.1  Per-domain city->file resolution (folded filename index, matching data_loader)
# --------------------------------------------------------------------------------------
_DOM_INDEX = {}


def _dom_index(subdir, suffix):
    """(exact, folded) filename indexes. EXACT first: 'San Jose' (US) and 'San José' (Costa Rica)
    are different cities that fold to the same key, and a folded-only index silently served one
    city's file for the other -- with a different winner per domain."""
    key = (subdir, suffix)
    if key not in _DOM_INDEX:
        exact, folded = {}, {}
        for path in glob.glob(os.path.join(V2, subdir, "*" + suffix)):
            base = os.path.basename(path)[: -len(suffix)]
            exact[base] = path
            folded.setdefault(_fold(base), path)
        _DOM_INDEX[key] = (exact, folded)
    return _DOM_INDEX[key]


def _resolve(subdir, suffix, city):
    exact, folded = _dom_index(subdir, suffix)
    return exact.get(city) or folded.get(_fold(city))


_DF_CACHE = {}


def _load(subdir, suffix, city):
    path = _resolve(subdir, suffix, city)
    if path is None:
        return None
    if path not in _DF_CACHE:
        _DF_CACHE[path] = pd.read_csv(path)
    return _DF_CACHE[path]


def load_hotels(city):
    return _load("hotel_data", "_hotel.csv", city)


def load_cars(city):
    return _load("car_data", "_rental_cars.csv", city)


def load_attr(city):
    return _load("attraction_data", "_attraction.csv", city)


def has_all_domains(city):
    return all(_resolve(sd, sf, city) is not None
               for sd, sf in (("hotel_data", "_hotel.csv"),
                              ("car_data", "_rental_cars.csv"),
                              ("attraction_data", "_attraction.csv")))


# --------------------------------------------------------------------------------------
# §1.2  Flight route graph  (directed; cheapest, plus cheapest at civil hours)
# --------------------------------------------------------------------------------------
CIVIL_DEPART_MIN = 6 * 60     # no reference leg departs before 06:00 if an alternative exists
CIVIL_ARRIVE_MAX = 22 * 60    # ...nor lands after 22:00


def _mins_str(s):
    """HH:MM -> minutes, or -1 when unparseable (so it can be compared in a pandas mask)."""
    try:
        h, m = str(s).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return -1


_ROUTE = None   # {(dep, arr): {price, flight_number, departure_time, arrival_time}}


def _build_routes():
    global _ROUTE
    if _ROUTE is not None:
        return
    df = pd.read_csv(FLIGHTS_CSV, keep_default_na=False,
                     usecols=["departure_city", "arrival_city", "price", "flight_number",
                              "departure_time", "arrival_time",
                              "departure_airport_latitude", "departure_airport_longitude",
                              "arrival_airport_latitude", "arrival_airport_longitude"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    def _rec(r):
        return {"price": float(r.price), "flight_number": r.flight_number,
                "departure_time": r.departure_time, "arrival_time": r.arrival_time,
                "dep_coord": (float(r.departure_airport_latitude), float(r.departure_airport_longitude)),
                "arr_coord": (float(r.arrival_airport_latitude), float(r.arrival_airport_longitude))}

    # Keep the cheapest row per directed pair AND the cheapest CIVIL-HOURS row. Picking purely on
    # price left 143 check-ins between 00:00 and 05:59 and 39 plans with under 5h of sleep, while
    # 81% of those legs had a daytime alternative on the same route for a median +$28/seat.
    dm = df["departure_time"].map(_mins_str)
    am = df["arrival_time"].map(_mins_str)
    civil = df[(dm >= CIVIL_DEPART_MIN) & (am <= CIVIL_ARRIVE_MAX) & (am > dm)]

    _ROUTE = {}
    for frame, key in ((df, "any"), (civil, "civil")):
        if frame.empty:
            continue
        best = frame.loc[frame.groupby(["departure_city", "arrival_city"])["price"].idxmin()]
        for r in best.itertuples(index=False):
            _ROUTE.setdefault((r.departure_city, r.arrival_city), {})[key] = _rec(r)


def flight_info(a, b, civil=False):
    """Cheapest direct a->b flight, or None. With `civil`, prefer the cheapest one that departs
    after 06:00 and lands by 22:00 on the same day, falling back to the outright cheapest when the
    route has no such service."""
    _build_routes()
    rec = _ROUTE.get((a, b))
    if not rec:
        return None
    if civil and "civil" in rec:
        return rec["civil"]
    return rec["any"]


def cheapest_flight(a, b):
    """(min_price, flight_number) of the cheapest direct a->b flight — the ABSOLUTE minimum, which
    is what `C_floor` must rest on, regardless of which flight the reference itinerary books."""
    fi = flight_info(a, b)
    return None if fi is None else (fi["price"], fi["flight_number"])


def route_graph():
    _build_routes()
    return set(_ROUTE.keys())


# --------------------------------------------------------------------------------------
# §1.3  Persona-aware cheapest options  (filters mirror the D1 scorer EXACTLY)
# --------------------------------------------------------------------------------------
def _grp(df, col, persona):
    """Boolean mask: persona ∈ the stringified-list group column."""
    def hit(x):
        try:
            return persona in ast.literal_eval(x)
        except Exception:
            return False
    return df[col].apply(hit)


def _filter_hotels(city, personas, req):
    """Hotels in `city` satisfying ALL personas + any req.star floor (df, or None if empty).
    luxury = amenities_group∋'luxury travelers' AND star>=5 (group tag alone ≠ 5★).
    foodie = rate_of_restaurant>=4 ONLY (foodie is never a hotel group tag)."""
    H = load_hotels(city)
    if H is None or H.empty:
        return None
    for p in personas:
        if p == "luxury travelers":
            H = H[_grp(H, "amenities_group", p) & (pd.to_numeric(H["star"], errors="coerce") >= 5)]
        elif p == "foodie":
            H = H[pd.to_numeric(H["rate_of_restaurant"], errors="coerce") >= 4]
        else:
            H = H[_grp(H, "amenities_group", p)]
        if H.empty:
            return None
    if req and req.get("star"):
        H = H[pd.to_numeric(H["star"], errors="coerce") >= float(req["star"])]
    if req and req.get("name"):
        # a named hotel constrains only the city it lives in; other stay cities keep their own pick
        sub = H[H["name"].astype(str) == str(req["name"])]
        if not sub.empty:
            H = sub
    return None if H.empty else H


_MEMO_HOTEL = {}


def pick_hotel(city, personas, req):
    """(price, name) of the cheapest satisfying hotel, else None. Memoized."""
    key = (city, tuple(sorted(personas)), (req or {}).get("star"), (req or {}).get("name"))
    if key in _MEMO_HOTEL:
        return _MEMO_HOTEL[key]
    H = _filter_hotels(city, personas, req)
    out = None
    if H is not None:
        prices = pd.to_numeric(H["price"], errors="coerce")
        tied = H[prices == prices.min()]
        r = tied.iloc[0]
        if len(tied) > 1:
            # STRICTLY cost-preserving tie-break: among the equally cheapest satisfying hotels,
            # take the one nearest the attractions this persona will actually visit. Price alone is
            # uncorrelated with location, so a pure cost-argmin lands at the 50th percentile of
            # distance-to-sights by construction.
            A = _filter_attr(city, personas)
            if A is not None and not A.empty:
                clat = float(pd.to_numeric(A["latitude"], errors="coerce").median())
                clon = float(pd.to_numeric(A["longitude"], errors="coerce").median())
                d = [_hav_xy((float(t.latitude), float(t.longitude)), (clat, clon))
                     for t in tied.itertuples()]
                r = tied.iloc[int(np.argmin(d))]
        out = (float(r["price"]), str(r["name"]))
    _MEMO_HOTEL[key] = out
    return out


def cheapest_hotel(city, personas, req):
    p = pick_hotel(city, personas, req)
    return None if p is None else p[0]


# personas the D1 scorer SKIPS for cars (excluded from denominator) -> no filter
_CAR_SKIP = {"fast-paced budget travel", "couples trip", "foodie"}


def _filter_cars(city, person_num, personas, req):
    """Cars in `city` seating >= max(person_num, req.capacity), matching req.car_type and all
    non-skip personas (df, or None if empty)."""
    C = load_cars(city)
    if C is None or C.empty:
        return None
    need = max(int(person_num), int((req or {}).get("capacity", 0) or 0))
    C = C[pd.to_numeric(C["capacity"], errors="coerce") >= need]
    if req and req.get("car_type"):
        C = C[C["car_type"].str.lower() == str(req["car_type"]).lower()]
    for p in personas:
        if p in _CAR_SKIP:
            continue
        C = C[_grp(C, "extra_services_group", p)]
        if C.empty:
            return None
    return None if C.empty else C


_MEMO_CAR = {}


def pick_car(city, person_num, personas, req):
    """(price_per_day, car_type) of the cheapest satisfying car, else None. Memoized."""
    key = (city, int(person_num), tuple(sorted(personas)),
           (req or {}).get("capacity"), (req or {}).get("car_type"))
    if key in _MEMO_CAR:
        return _MEMO_CAR[key]
    C = _filter_cars(city, person_num, personas, req)
    out = None
    if C is not None:
        r = C.loc[pd.to_numeric(C["price_per_day"], errors="coerce").idxmin()]
        out = (float(r["price_per_day"]), str(r["car_type"]))
    _MEMO_CAR[key] = out
    return out


def cheapest_car(city, person_num, personas, req):
    p = pick_car(city, person_num, personas, req)
    return None if p is None else p[0]


def _filter_attr(city, personas):
    """Attractions in `city` satisfying all personas (df, or None if empty).
    foodie also requires rate_of_restaurant>=3.5."""
    A = load_attr(city)
    if A is None or A.empty:
        return None
    if "foodie" in personas:
        A = A[pd.to_numeric(A["rate_of_restaurant"], errors="coerce") >= 3.5]
    for p in personas:
        A = A[_grp(A, "facilities_group", p)]
        if A.empty:
            return None
    return None if A.empty else A


_MEMO_TIX = {}


def pick_tickets(city, personas, n_att, allow_fewer=False):
    """The n_att cheapest satisfying attractions as [(price, name)], else None. With
    `allow_fewer`, returns however many exist (as long as at least A_SCHED do) instead of None —
    used when the day structure asks for more than the persona filter can supply. Memoized."""
    key = (city, tuple(sorted(personas)), int(n_att), bool(allow_fewer))
    if key in _MEMO_TIX:
        return _MEMO_TIX[key]
    A = _filter_attr(city, personas)
    out = None
    if A is not None:
        A = A.copy()
        A["_tp"] = pd.to_numeric(A["ticket_price"], errors="coerce")
        A = A.dropna(subset=["_tp"]).sort_values("_tp")
        # Collapse same-city CLONE families ("Vltava Walkway" / "Vltava Walkway No.2") to their
        # cheapest member. They are separate KB rows but the same place, and an itinerary that
        # visits both reads as a mistake — the agent prompt itself says not to repeat attractions.
        A = A.assign(_fam=A["attraction_name"].astype(str).str.replace(
            r"\s+No\.?\s*\d+\s*$", "", regex=True).str.strip().str.lower())
        A = A.drop_duplicates(subset="_fam", keep="first")
        take = n_att if len(A) >= n_att else (len(A) if allow_fewer else 0)
        if take >= (A_SCHED if allow_fewer else n_att) and take > 0:
            out = [(float(r["_tp"]), str(r["attraction_name"])) for _, r in A.head(take).iterrows()]
    _MEMO_TIX[key] = out
    return out


def _open_key(city, name):
    """(open, close) of an attraction, for scheduling the earlier-opening one first."""
    A = load_attr(city)
    row = A[A["attraction_name"] == name].iloc[0]
    o, cl = _parse_open(row["open_hours"])
    return (o if o is not None else 0, cl if cl is not None else 0)


def cheapest_tickets(city, personas, n_att):
    picks = pick_tickets(city, personas, n_att)
    return None if picks is None else [p for p, _ in picks]


def named_ticket_cost(meta, req):
    """Sum of ticket prices of any NAMED real attractions found in the stay cities, x person_num."""
    names = (req.get("attraction") or {}).get("name") or []
    if isinstance(names, str):
        names = [names]
    if not names:
        return 0.0
    total = 0.0
    for city in stay_cities_in_order(meta):
        A = load_attr(city)
        if A is None:
            continue
        for nm in names:
            m = A[A["attraction_name"].str.lower() == str(nm).lower()]
            if not m.empty:
                total += float(pd.to_numeric(m["ticket_price"], errors="coerce").min())
    return total * meta.person_num


# --------------------------------------------------------------------------------------
# §1.4  Itinerary legs + the two cost anchors (C_min_full / C_floor)
# --------------------------------------------------------------------------------------
A_SCHED = 2  # discretionary attractions booked per stay-city in the reference plan


def legs(meta):
    """[(D,c1),(c1,c2),...,(c_{k-1},c_k)] (+ (c_k,D) if round trip)."""
    D = (meta.req_flight or {}).get("departure_city")
    S = stay_cities_in_order(meta)
    seq = [D] + S
    out = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    if (meta.req_flight or {}).get("trip_type") == "round" and S:
        out.append((S[-1], D))
    return out


def flight_cost(meta):
    per = [cheapest_flight(a, b) for a, b in legs(meta)]
    if any(p is None for p in per):
        return None                       # route-infeasible
    return sum(p[0] for p in per) * meta.person_num


def reference_cost(meta, personas, req, R):
    """(cost, schedule) of the reference plan. The SINGLE source of truth for C_min: the number of
    attractions now depends on the day structure, so the only way the cost model and the shipped
    gold can never disagree is to price the actual schedule with the actual scorer."""
    sch = build_schedule(meta, personas, req, R)
    if sch is None:
        return None, None
    return scorer().compute_total_cost(scorer().extract_plan_data(sch), meta), sch


def C_min_full(meta, personas, req, R):
    """Cost of the intended persona-satisfying reference plan == the scorer's bill of it."""
    return reference_cost(meta, personas, req, R)[0]


def C_floor(meta, req, R):
    """True hard lower bound of ANY valid plan: flights + per-city lodging (persona-BLIND) +
    car iff required (persona-blind) + only a NAMED attraction (discretionary ones droppable).
    Persona premium and discretionary attractions are excluded -> a real lower bound."""
    cities, idx, nights, cardays, _bn, _bd = city_maps(meta)
    fc = flight_cost(meta)
    if fc is None:
        return None
    hc = 0.0
    for c in cities:
        p = cheapest_hotel(c, [], {})     # persona-blind cheapest room
        if p is None:
            return None
        hc += p * meta.rooms_count * nights[idx[_fold(c)]]
    cc = 0.0
    if "car" in R:
        for c in cities:
            p = cheapest_car(c, meta.person_num, [], {})
            if p is None:
                return None
            cc += p * cardays[idx[_fold(c)]]
    tc = named_ticket_cost(meta, req)     # only forced named attractions count
    return fc + hc + cc + tc


# --------------------------------------------------------------------------------------
# §1.5  Reference-plan builder + self-verify against the REAL scorer
# --------------------------------------------------------------------------------------
def build_reference_plan(meta, personas, req, R, feasible=True):
    """The argmin reference plan (entities named) whose scorer bill == C_min_full.
    Returns a plan_data dict the scorer accepts, or None if any component is infeasible."""
    cities = stay_cities_in_order(meta)
    flights = []
    for a, b in legs(meta):
        cf = cheapest_flight(a, b)
        if cf is None:
            return None
        flights.append({"flight_number": cf[1], "departure_city": a, "arrival_city": b})
    hotels = []
    for c in cities:
        ph = pick_hotel(c, personas, req.get("hotel", {}))
        if ph is None:
            return None
        hotels.append({"name": ph[1], "city": c})
    cars = []
    if "car" in R:
        for c in cities:
            pc = pick_car(c, meta.person_num, personas, req.get("car", {}))
            if pc is None:
                return None
            # carry price_per_day so the scorer's get_car_price(claimed=...) disambiguates to THIS
            # car among the 3 same-(city,type) cars (else it defaults to iloc[0] and over/under-bills).
            cars.append({"car_type": pc[1], "city": c, "price_per_day": pc[0]})
    attractions = []
    if "attraction" in R:
        for c in cities:
            pt = pick_tickets(c, personas, A_SCHED)
            if pt is None:
                return None
            for _, nm in pt:
                attractions.append({"name": nm, "city": c})
    return {"flights": flights, "hotels": hotels, "cars": cars,
            "attractions": attractions, "is_feasible": feasible}


_SCORER = None


def scorer():
    """Lazily build the authoritative TravelPlanScorer over the v2 sandbox DB."""
    global _SCORER
    if _SCORER is None:
        from scoring import TravelPlanScorer
        from data_loader import get_sandbox_db
        _SCORER = TravelPlanScorer(get_sandbox_db())
    return _SCORER


def scorer_cost(plan, meta):
    return scorer().compute_total_cost(plan, meta)


# ======================================================================================
# §2  SAMPLING PIPELINE  (5 stages -> 800 rows + gold reference plans)
# ======================================================================================

PERSONAS = ['with children', 'road trip', 'elderly travelers', 'business travelers', 'with pets',
            'nightlife enthusiast', 'disabled traveler', 'fast-paced budget travel', 'couples trip',
            'solo women', 'luxury travelers', 'foodie', 'photography']
_PID = {i + 1: p for i, p in enumerate(PERSONAS)}

# The v1 rule set (19 hand-written pairs; Text/appendix.tex:312-319). RETIRED 2026-07-19 —
# kept only so the paper rewrite can cite what changed. An audit of all 105,335 v2 KB
# resources found the KB tags BOTH members of every "conflicting" pair on 2,327-6,269
# resources, versus a 5,529 median for the pairs the same rules called legal: the asserted
# conflicts were statistically indistinguishable from the permitted ones. Several also
# encoded age/disability stereotypes ("nightlife enthusiast" x "disabled traveler",
# "elderly travelers" x "solo women"), which is a reviewer/ethics liability, and the set was
# internally inconsistent (couples x solo women conflicts, but couples x with children does not).
LEGACY_CONFLICT_PAIRS_V1 = {frozenset((_PID[a], _PID[b])) for a, b in [
    (1, 6), (1, 8), (1, 4), (1, 10), (3, 6), (3, 8), (3, 10), (4, 5), (4, 8), (4, 9), (4, 13),
    (5, 6), (5, 8), (5, 11), (5, 12), (6, 7), (7, 8), (8, 11), (9, 10)]}

# ---- v2 persona compatibility: ONLY logically-impossible combinations are excluded. ----
# (a) Party-size contradictions, DERIVED from person_num rather than hand-listed.
PARTY_SIZE = {
    'solo women':    (1, 1),   # exactly one traveler
    'couples trip':  (2, 2),   # a couple is exactly two — (2,5) produced "5 travelers ... for
                               # couples trip" in 52 of 64 couples queries, 35 of them an ODD party
                               # size that cannot be composed of couples at all, in the user-visible
                               # query text
    'with children': (2, 5),   # at least one adult plus a child
}
# (b) Opposite ends of a single ordinal axis (spend level). This is the only true same-axis pair.
AXIS_CONFLICTS = {frozenset(('fast-paced budget travel', 'luxury travelers'))}


def party_size_range(personas):
    """Feasible [lo, hi] party size implied by the personas; lo > hi means contradictory."""
    lo, hi = 1, 5
    for p in personas:
        a, b = PARTY_SIZE.get(p, (1, 5))
        lo, hi = max(lo, a), min(hi, b)
    return lo, hi

FAKE_HOTELS = ['Grand Aurora Palace Hotel', 'Nimbus Skyline Suites', 'Velvet Harbour Grand',
               'The Obsidian Crown Hotel', 'Celestia Riverside Lodge', 'Marble Lantern Inn',
               'Azure Meridian Hotel', 'The Gilded Compass Hotel']
FAKE_ATTR = ['Museum of Forgotten Tides', 'The Whispering Fern Conservatory', 'Obsidian Spire Overlook',
             'Gallery of Vanished Constellations', 'The Sunken Lantern Grotto', 'Meridian Clockwork Gardens']
FAKE_CAR_TYPES = ['Hovercraft', 'Amphibious Cruiser', 'Monorail Pod', 'Gyrocar']

REAL_CAR_TYPES_LOWER = {t.lower() for t in CAR_TYPES}


def _conflict_free(ps):
    """True unless the set is logically impossible: a same-axis contradiction, or a party-size
    requirement with an empty feasible range."""
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            if frozenset((ps[i], ps[j])) in AXIS_CONFLICTS:
                return False
    lo, hi = party_size_range(ps)
    return lo <= hi


# ---- adjacency + usable-city universe (cached) ----
_USABLE = None
_OUT = None


def usable_cities():
    global _USABLE
    if _USABLE is None:
        G = route_graph()
        cities = {a for a, _ in G} | {b for _, b in G}
        _USABLE = sorted(c for c in cities if has_all_domains(c))
    return _USABLE


def out_neighbors(city):
    global _OUT
    if _OUT is None:
        us = set(usable_cities())
        _OUT = {}
        for (a, b) in route_graph():
            if a in us and b in us:
                _OUT.setdefault(a, []).append(b)
    return _OUT.get(city, [])


def _choice(rng, seq):
    return seq[int(rng.integers(len(seq)))]


# ---- geographic coherence -------------------------------------------------------------
# Flight connectivity alone allowed absurd itineraries (Perth -> George Town, Atlanta =
# 18,102 km in six days). Real multi-city travel is REGIONAL: you may fly long-haul to a
# region, then tour within it. So we cap the spread among the STAY cities only; the
# departure -> first-stop leg is left unconstrained.
MAX_STAY_SPREAD_KM = 3000.0
_COORD = None


def _coords():
    global _COORD
    if _COORD is None:
        df = pd.read_csv(FLIGHTS_CSV, keep_default_na=False,
                         usecols=["departure_city", "departure_airport_latitude", "departure_airport_longitude",
                                  "arrival_city", "arrival_airport_latitude", "arrival_airport_longitude"])
        c = {}
        for a, la, lo in zip(df.departure_city, df.departure_airport_latitude, df.departure_airport_longitude):
            c.setdefault(a, (float(la), float(lo)))
        for a, la, lo in zip(df.arrival_city, df.arrival_airport_latitude, df.arrival_airport_longitude):
            c.setdefault(a, (float(la), float(lo)))
        _COORD = c
    return _COORD


def _km(a, b):
    C = _coords()
    if a not in C or b not in C:
        return 0.0
    (la1, lo1), (la2, lo2) = C[a], C[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(x))


def stay_spread_ok(cities):
    """Stay cities must lie within one region (max pairwise distance <= MAX_STAY_SPREAD_KM)."""
    return all(_km(a, b) <= MAX_STAY_SPREAD_KM
               for i, a in enumerate(cities) for b in cities[i + 1:])


# ---- Stage 2 helpers: itinerary sampling ----
def sample_connected(rng, k, round_trip):
    """(D, [c1..ck]) with every leg in G (+ return leg if round_trip), else None."""
    us = usable_cities()
    G = route_graph()
    for _ in range(300):
        D = _choice(rng, us)
        chain, cur, ok = [], D, True
        for _step in range(k):
            nbrs = [b for b in out_neighbors(cur) if b != D and b not in chain]
            if not nbrs:
                ok = False
                break
            nxt = _choice(rng, nbrs)
            chain.append(nxt)
            cur = nxt
        if not ok:
            continue
        if not stay_spread_ok(chain):          # regional touring only
            continue
        if round_trip and (chain[-1], D) not in G:
            continue
        return D, _best_city_order(D, chain, round_trip, G)
    return None


def _best_city_order(D, chain, round_trip, G):
    """Emit the stay cities in the SHORTEST order the direct-flight graph actually allows.

    `sample_connected` walks the route graph uniformly at random, which made 82 of 282 multi-city
    trips fly >5% further than necessary. The order is baked into `req_flight.arrival_city` AND into
    the query text, so the benchmark was forcing every model to reproduce the zig-zag and then
    scoring it as correct. Restricted to FLYABLE permutations — reordering by pure geometry would
    make a large share of tasks route-infeasible (the flight graph is only 7.9% dense)."""
    if len(chain) < 2:
        return chain
    from itertools import permutations
    best, best_d = chain, None
    for perm in permutations(chain):
        legs = [(D, perm[0])] + [(perm[i], perm[i + 1]) for i in range(len(perm) - 1)]
        if round_trip:
            legs.append((perm[-1], D))
        if any(l not in G for l in legs):
            continue
        d = sum(_km(a, b) for a, b in legs)
        if best_d is None or d < best_d - 1e-9:
            best, best_d = list(perm), d
    return best


def sample_no_route(rng, k, round_trip):
    """Itinerary where D->c1 has NO direct flight while EVERY other required leg does exist — so
    exactly one leg is missing. Supports round trips (the return ck->D must exist), which keeps
    `trip_type` uncorrelated with the impossible label: forcing this class to one-way used to make
    one-way tasks 50.8% impossible vs 25.3% for round trips, a free giveaway signal."""
    us = usable_cities()
    G = route_graph()
    for _ in range(600):
        if k == 1:
            chain = [_choice(rng, us)]
        else:
            base = sample_connected(rng, k, False)
            if base is None:
                continue
            chain = base[1]
        c1, ck = chain[0], chain[-1]
        for _ in range(60):
            D = _choice(rng, us)
            if D == c1 or D in chain or (D, c1) in G:
                continue
            if round_trip and (ck, D) not in G:
                continue
            return D, chain
    return None


# ---- fake-entity generation (verified absent) ----
def fake_hotel_name(rng, city):
    H = load_hotels(city)
    existing = set(H['name'].astype(str).str.lower()) if H is not None else set()
    for _ in range(60):
        nm = f"{_choice(rng, FAKE_HOTELS)} {int(rng.integers(2, 99))}"
        if nm.lower() not in existing:
            return nm
    return f"{_choice(rng, FAKE_HOTELS)} {int(rng.integers(100, 9999))}"


def fake_attraction_name(rng, city):
    A = load_attr(city)
    existing = set(A['attraction_name'].astype(str).str.lower()) if A is not None else set()
    for _ in range(60):
        nm = f"{_choice(rng, FAKE_ATTR)} {int(rng.integers(2, 99))}"
        if nm.lower() not in existing:
            return nm
    return f"{_choice(rng, FAKE_ATTR)} {int(rng.integers(100, 9999))}"


# ---- Stage 3: personas (feasibility-aware greedy: guarantees a satisfying hotel + A_SCHED
#      attractions exist in EVERY stay city, so C_min_full won't fail on personas) ----
def sample_personas(rng, count, cities):
    """`count` compatible personas jointly satisfiable in every stay city. Party size is NOT an
    input: it is derived from the chosen set afterwards via `party_size_range`. Greedily adds a
    persona only while every city keeps a hotel + A_SCHED attractions satisfying the whole set."""
    if count == 0:
        return []
    for _attempt in range(40):
        pool = list(PERSONAS)
        rng.shuffle(pool)
        chosen = []
        for p in pool:
            if len(chosen) == count:
                break
            cand = chosen + [p]
            if not _conflict_free(cand):
                continue
            if all(cheapest_hotel(c, cand, {}) is not None
                   and cheapest_tickets(c, cand, A_SCHED) is not None for c in cities):
                chosen = cand
        if len(chosen) == count:
            return chosen
    return None


# ---- persona binding check (H7): >=1 persona strictly price-binding in >=1 stay city ----
def _persona_binds(cities, personas, req):
    """Does at least one persona strictly raise the cheapest hotel price somewhere? Evaluated with
    any explicit `name` pin STRIPPED: whether a persona is meaningful is a property of the city's
    inventory, not of whether this particular query happens to name a hotel. (Leaving the pin in
    made every named-hotel task look non-binding -- it pins both sides to the same hotel -- which
    rerolled them into oblivion and skewed the hotel/attraction naming split.)"""
    if not personas:
        return True
    hreq = {k: v for k, v in (req.get('hotel') or {}).items() if k != 'name'}
    for c in cities:
        blind = cheapest_hotel(c, [], hreq)
        tag = cheapest_hotel(c, personas, hreq)
        if blind is not None and tag is not None and tag > blind + 1e-9:
            return True
    return False


# ---- Stage 4: required domains ----
def choose_domains(rng, personas):
    """Returns (required_domains, car_mode). Every car requirement must be INFERABLE by the agent:
      'implicit' -- the `road trip` persona IS the signal; the text stays silent on purpose
                    (this is exactly what the implicit-need dimension is meant to test);
      'explicit' -- no persona implies a car, so the query text MUST state the requirement;
      None       -- no car at all.
    (Bug fixed 2026-07-19: a bare 35% coin-flip used to add `req_car` with no signal anywhere,
    leaving 255 tasks demanding a car the query never mentioned and inflating their budgets.)"""
    R = {'flight', 'hotel', 'attraction'}
    if 'road trip' in personas:
        R.add('car')
        return R, 'implicit'
    if rng.random() < 0.25:
        R.add('car')
        return R, 'explicit'
    return R, None


# ---- budget bands (A.4) ----
def _band(rng, lo, hi):
    return lo + float(rng.random()) * (hi - lo)


def _round_meta(req_flight, days, person_num, cities_count):
    import types
    return types.SimpleNamespace(req_flight=req_flight, days=days, person_num=person_num,
                                 rooms_count=rooms_for(person_num), cities_count=cities_count)


# ---- templates (pure, no order words). {tt} is a full noun phrase (no trailing "trip"). ----
def _trip_phrase(rng, tt):
    if tt == "round":
        return _choice(rng, ["round-trip journey", "round-trip getaway", "round-trip vacation", "round trip"])
    return _choice(rng, ["one-way trip", "one-way journey", "one-way getaway"])


_TPL = [
    "Please help plan a {tt} for {pp} from {dep} to {arr}. We'll stay {days} days and need {rooms} hotel room(s){per}. Our budget is {bud} USD.{ent}",
    "I'm arranging a {tt} from {dep} to {arr} for {pp}. We'd like to stay {days} days in {rooms} hotel room(s){per}, keeping the cost within {bud} USD.{ent}",
    "Looking for a {tt} from {dep} to {arr}. There are {ppn} of us staying {days} days in {rooms} room(s){per}, with a budget of {bud} USD.{ent}",
    "Could you organise a {tt} from {dep} to {arr} for {pp}? Plan for {days} days and {rooms} hotel room(s){per}. Total budget is {bud} USD.{ent}",
    "We want a {tt} from {dep} to {arr} for {pp}: {days} days, {rooms} hotel room(s){per}, budget {bud} USD.{ent}",
    "Help me put together a {tt} from {dep} to {arr}. {pp}, {days} days, {rooms} hotel room(s){per}, and we can spend up to {bud} USD.{ent}",
    "I need a {tt} from {dep} to {arr} for {pp}, lasting {days} days with {rooms} hotel room(s){per}. Please keep it under {bud} USD.{ent}",
    "Can you sort out a {tt} from {dep} to {arr}? It's for {pp}, {days} days, {rooms} hotel room(s){per}, max spend {bud} USD.{ent}",
    "My group of {ppn} is taking a {tt} from {dep} to {arr}. We need {days} days and {rooms} hotel room(s){per}, all within {bud} USD.{ent}",
    "Booking a {tt} from {dep} to {arr} for {pp}. Duration {days} days, {rooms} hotel room(s){per}. Budget ceiling: {bud} USD.{ent}",
    "We're planning a {tt} from {dep} to {arr}. {pp} for {days} days, {rooms} hotel room(s){per}, spending no more than {bud} USD.{ent}",
    "Any chance you could plan a {tt} from {dep} to {arr}? {pp}, {days} days away, {rooms} hotel room(s){per}, budget {bud} USD.{ent}",
    "Trip request: a {tt} from {dep} to {arr} for {pp}, {days} days, {rooms} hotel room(s){per}. Budget is {bud} USD.{ent}",
    "I'd like to book a {tt} from {dep} to {arr}. Travelling {pp} for {days} days, needing {rooms} hotel room(s){per}, under {bud} USD.{ent}",
    "Set up a {tt} from {dep} to {arr} please. {pp}, {days} days, {rooms} hotel room(s){per}, total not exceeding {bud} USD.{ent}",
    "Hoping to arrange a {tt} from {dep} to {arr} for {pp}. Staying {days} days in {rooms} hotel room(s){per}, budget {bud} USD.{ent}",
    "What can you do for a {tt} from {dep} to {arr}? {pp}, {days} days, {rooms} hotel room(s){per}, {bud} USD to spend.{ent}",
    "Assist me with a {tt} from {dep} to {arr}. Party of {ppn}, {days} days, {rooms} hotel room(s){per}, capped at {bud} USD.{ent}",
    "Planning a {tt} from {dep} to {arr} for {pp} over {days} days, with {rooms} hotel room(s){per}. We can afford {bud} USD.{ent}",
    "A {tt} from {dep} to {arr} is what we need: {pp}, {days} days, {rooms} hotel room(s){per}, budget {bud} USD.{ent}",
    "Organising travel for {pp}: a {tt} from {dep} to {arr}, {days} days, {rooms} hotel room(s){per}, within {bud} USD.{ent}",
    "Could I get a {tt} from {dep} to {arr} arranged? {pp} travelling {days} days, {rooms} hotel room(s){per}, budget {bud} USD.{ent}",
    "Requesting a {tt} from {dep} to {arr}. We are {ppn}, staying {days} days, needing {rooms} hotel room(s){per}. Spending limit {bud} USD.{ent}",
    "Put together a {tt} from {dep} to {arr} for {pp}, please. {days} days, {rooms} hotel room(s){per}, no more than {bud} USD.{ent}",
]


def _people_phrase(n):
    return "1 traveler" if n == 1 else f"{n} travelers"


def _arr_phrase(cities):
    if len(cities) == 1:
        return cities[0]
    return ", ".join(cities[:-1]) + " and " + cities[-1]


def _persona_phrase(personas):
    if not personas:
        return ""
    return " for " + (personas[0] if len(personas) == 1
                      else ", ".join(personas[:-1]) + " and " + personas[-1])


def _car_phrase(car_mode, person_num):
    """Only an 'explicit' car requirement is verbalised; 'implicit' relies on the road-trip persona."""
    if car_mode != 'explicit':
        return ""
    return f" We'll also need a rental car seating {max(4, person_num)}."


_DIRECT_TPL = [
    " We only fly direct — no connecting itineraries.",
    " Direct flights only, please; we won't take connections.",
    " One condition: every flight must be direct, no layovers.",
    " We don't do layovers, so keep every leg a direct flight.",
    " Please keep it to direct flights only — no stopovers.",
]


# The city ORDER is a stated constraint: required_legs() derives the itinerary as
# departure -> city1 -> ... -> cityN in the order the request lists them, and 4 of the 89
# route-impossible tasks were solvable by permuting that list, so the text has to say so.
# Deliberately worded to avoid data_loader's is_ordered keywords ("first/then/after/..."),
# which would switch D2 from coverage to an LCS order score — a different metric.
ORDER_CLAUSE = " Please visit the cities in the order listed."


def _direct_phrase(rng):
    """The direct-flight constraint, stated on EVERY task.

    This is what makes the no-route class provably infeasible rather than a hidden benchmark
    convention: the flight graph is strongly connected (0 unreachable pairs out of 140,250), so
    "no route" is only true relative to a stated no-connections requirement. Because the clause
    appears on 100% of tasks — feasible ones are built from the direct-flight graph and already
    satisfy it — its presence leaks nothing about feasibility.
    """
    return _choice(rng, _DIRECT_TPL)


def _entity_phrase(rng, kind, name, etype=None):
    """Name the mandated entity AND say what kind of thing it is.

    "We must book X only" never said whether X was a hotel or an attraction, so an agent had to
    guess which index to search. In the pilot that ambiguity alone produced false refusals on 10%
    of tasks: the agent searched hotels for what was actually an attraction, got nothing back, and
    concluded the entity did not exist. The task is supposed to test whether the agent VERIFIES a
    named entity against the KB, not whether it guesses the right table.

    Stating the type leaks nothing: the real-entity and fake-entity classes draw hotels vs
    attractions from the SAME 70/30 split, so "names an attraction" carries no feasibility signal.
    """
    if kind is None:
        return ""
    if etype == 'attraction':
        if kind == 'want':
            return f" We'd like to visit the attraction {name}."
        return f" We must visit the attraction {name} — nothing else will do."
    if etype == 'car':
        if kind == 'want':
            return f" We'd like to rent a {name}."
        return f" We must rent a {name} — nothing else will do."
    # default: hotel
    if kind == 'want':
        return f" We'd like to stay at the hotel {name}."
    return f" We must stay at the hotel {name} — nothing else will do."


def render_query(rng, tt, cities, dep, days, rooms, person_num, budget, personas,
                 entity_kind, entity_name, car_mode=None, entity_type=None):
    tpl = _choice(rng, _TPL)
    # trailing clauses share the {ent} slot: car requirement, any mandatory entity, then the
    # direct-flight condition (present on every task, so it carries no feasibility signal)
    extras = (_car_phrase(car_mode, person_num)
              + _entity_phrase(rng, entity_kind, entity_name, entity_type)
              + _direct_phrase(rng))
    txt = tpl.format(tt=_trip_phrase(rng, tt),
                     pp=_people_phrase(person_num), ppn=person_num,
                     dep=dep, arr=_arr_phrase(cities), days=days, rooms=rooms,
                     bud=int(round(budget)), per=_persona_phrase(personas),
                     ent=extras)
    return " ".join(txt.split())   # collapse any double spaces


# ---- row assembly ----
def _row(query, personas, budget, dep, cities, tt, k, person_num, rooms, days, impossible,
         balance, req_hotel, req_car, req_attraction):
    req_flight = {'departure_city': dep, 'arrival_city': cities, 'trip_type': tt, 'price': ''}
    hard = {'person_num': person_num, 'budget': str(int(round(budget))), 'days': days}
    return {
        'query': query, 'level': 'mixed',
        'implicit_keywords': repr(personas),
        'hard_constraints': repr(hard),
        'req_flight': repr(req_flight), 'req_car': repr(req_car),
        'req_hotel': repr(req_hotel), 'req_attraction': repr(req_attraction),
        'cities_count': repr({'cities_count': k}),
        'implicit_keywords_count': len(personas),
        'balance': balance, 'impossible': 1.0 if impossible else 0.0,
        'total_score': '',
    }


COLUMNS = ['query', 'level', 'implicit_keywords', 'hard_constraints', 'req_flight', 'req_car',
           'req_hotel', 'req_attraction', 'cities_count', 'implicit_keywords_count', 'balance',
           'impossible', 'total_score']


def build_task(rng, k, cls, persona_count):
    """One attempt to build a task of (cities_count=k, class=cls, persona_count). Returns
    (row_dict, reference_plan) on success (self-verified), or None to reroll."""
    # trip_type is drawn the SAME way for every class, so it leaks nothing about feasibility
    round_trip = rng.random() < 0.75
    tt = 'round' if round_trip else 'one_way'
    days = 3 * k

    # --- itinerary ---
    if cls == 'imp_route':
        itin = sample_no_route(rng, k, round_trip)
    else:
        itin = sample_connected(rng, k, round_trip)
    if itin is None:
        return None
    dep, cities = itin

    # --- personas (feasibility-aware over the chosen stay cities) ---
    personas = sample_personas(rng, persona_count, cities)
    if personas is None:
        return None
    # party size is DERIVED from the personas -- the only demographic constraint we encode
    _lo, _hi = party_size_range(personas)
    person_num = int(rng.integers(_lo, _hi + 1))
    rooms = rooms_for(person_num)

    req_flight = {'departure_city': dep, 'arrival_city': cities, 'trip_type': tt}
    meta = _round_meta(req_flight, days, person_num, k)
    R, car_mode = choose_domains(rng, personas)

    # req_* skeleton (H5: req_hotel ALWAYS present w/ rooms_count; H12: req_attraction present)
    req_hotel = {'rooms_count': str(rooms)}
    req_attraction = {'name': []}
    req_car = {}
    if 'car' in R:
        # general car requirement is capacity-only (type-flexible) so multi-city stays feasible;
        # a pinned car_type is reserved for the D0-key-car subset / entity-nonexistent-car.
        req_car = {'capacity': str(max(4, person_num))}
    # Name a REAL entity on a share of the non-fake classes. Without this, "the query names a
    # specific hotel/attraction" was 100% predictive of `impossible` (the fake-entity class was the
    # only one that ever named anything) -- a free giveaway -- and D0-key was never exercised on a
    # solvable task. We always name the entity the reference plan ALREADY books, so cost, D1 and the
    # budget are unchanged; only the D0-key constraint is added. The wording is the same mandatory
    # phrasing the fake-entity class uses, so the wording leaks nothing either.
    # EVERY task names an entity. The entity-nonexistent class must name one by definition and is
    # exactly 1/3 of the impossible pool, so any naming rate p < 1 leaves "the query names a hotel"
    # predictive of infeasibility ((89+178p)/(89+711p) = 0.334 solves to p ~ 1.0). Naming everywhere
    # also makes that class test the RIGHT thing: the agent has to VERIFY the name against the KB
    # instead of pattern-matching on "a proper noun appeared".
    entity_kind, entity_name, entity_type = None, None, None
    if cls != 'imp_entity':
        if rng.random() < 0.70:
            ph = pick_hotel(cities[0], personas, req_hotel)
            if ph is not None:
                entity_kind, entity_name, entity_type = 'only', ph[1], 'hotel'
                req_hotel['name'] = ph[1]
        else:
            pt = pick_tickets(cities[0], personas, A_SCHED)
            if pt:
                entity_kind, entity_name, entity_type = 'only', pt[0][1], 'attraction'
                req_attraction = {'name': [pt[0][1]]}
    req = {'hotel': req_hotel, 'car': req_car, 'attraction': req_attraction}

    balance = ''

    # ---------------- class-specific budget + labels ----------------
    if cls in ('solv_tight', 'solv_loose', 'imp_budget'):
        # binding persona check (H7)
        if not _persona_binds(cities, personas, req):
            return None
        cfloor = C_floor(meta, req, R)
        cmin = C_min_full(meta, personas, req, R)
        if cfloor is None or cmin is None:
            return None
        if cls == 'solv_tight':
            budget = float(np.ceil(cmin * _band(rng, 1.02, 1.10)))
            impossible, balance = False, 1.0
        elif cls == 'solv_loose':
            budget = float(np.ceil(cmin * _band(rng, 1.35, 1.80)))
            impossible, balance = False, 0.0
        else:  # imp_budget
            budget = float(np.floor(cfloor * _band(rng, 0.55, 0.92)))
            budget = min(budget, cfloor - max(50.0, 0.05 * cfloor))
            if budget <= 0:
                return None
            impossible = True

    elif cls == 'imp_route':
        # generous, non-binding budget from a median-flight stand-in for the missing leg
        cmin_realish = _est_cost_no_route(meta, personas, req, R)
        if cmin_realish is None:
            return None
        budget = float(np.ceil(cmin_realish * 1.4))
        impossible = True

    elif cls == 'imp_entity':
        cmin = C_min_full(meta, personas, req, R)   # real-entity cost estimate
        if cmin is None:
            return None
        budget = float(np.ceil(cmin * 1.4))
        impossible = True
        # inject one fake entity, using the SAME 70/30 hotel/attraction split the real-entity
        # naming uses, so neither "names a hotel" nor "names an attraction" carries any signal.
        # (The fake-car-type variant was dropped: score_d0_source and the D4 hallucination check
        # do not verify cars, so it was a weak vehicle anyway.)
        if rng.random() < 0.70:
            entity_kind, entity_name, entity_type = 'only', fake_hotel_name(rng, cities[0]), 'hotel'
            req_hotel = {'rooms_count': str(rooms), 'name': entity_name}
        else:
            entity_kind, entity_name, entity_type = 'only', fake_attraction_name(rng, cities[0]), 'attraction'
            req_attraction = {'name': [entity_name]}
    else:
        return None

    # ---------------- text ----------------
    query = render_query(rng, tt, cities, dep, days, rooms, person_num, budget, personas,
                         entity_kind, entity_name, car_mode, entity_type)

    row = _row(query, personas, budget, dep, cities, tt, k, person_num, rooms, days,
               impossible, balance, req_hotel, req_car, req_attraction)

    # ---------------- self-verify against the REAL scorer ----------------
    if not _verify(cls, meta, personas, req, R, budget, cities, entity_name):
        return None

    # ---------------- gold answer ----------------
    # solvable -> a schedule-complete plan in the agent's own output schema (so it can be scored
    # on EVERY dimension, including B2/B3); impossible -> the correct answer is a structured refusal.
    if impossible:
        gold = {'is_feasible': False, 'refusal_reason': _refusal_reason(cls, meta, entity_name),
                'plan': {}}
    else:
        gold = build_schedule(meta, personas, req, R)
        if gold is None:
            return None
        # A stay city the traveller never sees anything in makes the gold a flight-and-hotel bundle
        # rather than a travel plan. Reroll instead of shipping it. This also weeds out the
        # degenerate 3-day intercontinental tasks whose single stay day is consumed by a
        # date-line-crossing arrival (e.g. San Francisco->Zurich 19:37->15:22, check-in 16:52,
        # every eligible attraction shut by 17:00).
        for _c in stay_cities_in_order(meta):
            if not any(a.get("city") == _c
                       for d in gold["plan"].values() for a in d.get("attractions", [])):
                return None
        # A named entity the scheduler could not place would cost D0-key on an answer that is
        # supposed to be perfect (it happens when the attraction's opening window does not
        # intersect any of this itinerary's slots). Reroll rather than ship it.
        _planned = {a.get("name") for d in gold["plan"].values() for a in d.get("attractions", [])}
        if any(nm not in _planned for nm in (req_attraction.get("name") or [])):
            return None
        _hn = req_hotel.get("name")
        if _hn and _hn not in {d["hotel"]["name"] for d in gold["plan"].values() if d.get("hotel")}:
            return None
        _pd = scorer().extract_plan_data(gold)
        if scorer().compute_total_cost(_pd, meta) > budget + 1e-6:
            return None
    return row, gold


def _refusal_reason(cls, meta, entity_name):
    dep = (meta.req_flight or {}).get('departure_city')
    arr = stay_cities_in_order(meta)
    if cls == 'imp_budget':
        return ("The stated budget is below the minimum achievable cost of any valid itinerary "
                "for this trip.")
    if cls == 'imp_route':
        # "No flight connects A to B" was FALSE: the route graph is strongly connected, so a
        # connecting itinerary always exists. What is true — and what the traveller ruled out — is
        # that no DIRECT flight serves the leg.
        miss = [f"{a} to {b}" for a, b in legs(meta) if cheapest_flight(a, b) is None]
        leg = miss[0] if miss else f'{dep} to {arr[0]}'
        return (f"No direct flight serves {leg}, and the traveller rules out connecting "
                f"itineraries, so the trip cannot be booked as specified.")
    if cls == 'imp_entity':
        return f"The requested '{entity_name}' does not exist in the knowledge base."
    return "The request cannot be satisfied."


def _est_cost_no_route(meta, personas, req, R):
    """Cost estimate for a no-route trip: median flight price stands in for the missing leg."""
    _build_routes()
    med = float(np.median([v["any"]["price"] for v in _ROUTE.values()]))
    n_legs = len(legs(meta))
    fc = med * n_legs * meta.person_num
    cities, idx, nights, cardays, *_ = city_maps(meta)
    hc = 0.0
    for c in cities:
        p = cheapest_hotel(c, personas, req.get('hotel', {}))
        if p is None:
            return None
        hc += p * meta.rooms_count * nights[idx[_fold(c)]]
    return fc + hc


def _verify(cls, meta, personas, req, R, budget, cities, entity_name):
    """Assert the label matches the authoritative scorer / KB before shipping the row."""
    if cls in ('solv_tight', 'solv_loose'):
        plan = build_reference_plan(meta, personas, req, R, feasible=True)
        if plan is None:
            return False
        return scorer_cost(plan, meta) <= budget + 1e-6
    if cls == 'imp_budget':
        return C_floor(meta, req, R) > budget + 1e-6
    if cls == 'imp_route':
        return any(cheapest_flight(a, b) is None for a, b in legs(meta))
    if cls == 'imp_entity':
        db = scorer().db
        # fake hotel / attraction must be absent; fake car_type must be outside the 9-type vocab
        if str(entity_name).lower() in REAL_CAR_TYPES_LOWER:
            return False
        for c in cities:
            H = load_hotels(c)
            A = load_attr(c)
            if H is not None and str(entity_name).lower() in set(H['name'].astype(str).str.lower()):
                return False
            if A is not None and str(entity_name).lower() in set(A['attraction_name'].astype(str).str.lower()):
                return False
        return True
    return False


def build_quota_specs(n=800):
    """Stage 1: 800 (cities_count, class, persona_count) specs by shuffled stratified quotas."""
    def spread(counts):
        out = []
        for val, c in counts:
            out += [val] * c
        return out
    kk = spread([(1, 360), (2, 240), (3, 200)])
    cls = spread([('solv_tight', 267), ('solv_loose', 266),
                  ('imp_budget', 89), ('imp_route', 89), ('imp_entity', 89)])
    pc = spread([(0, 120), (1, 280), (2, 240), (3, 160)])
    assert len(kk) == len(cls) == len(pc) == n, (len(kk), len(cls), len(pc))
    rng = np.random.default_rng(20260719)
    rng.shuffle(kk)
    rng.shuffle(cls)
    rng.shuffle(pc)
    return list(zip(kk, cls, pc))


def generate(n=800, out_csv='trek_queries.csv', out_gold='trek_reference_plans.jsonl', seed=42):
    specs = build_quota_specs(n)
    rng = np.random.default_rng(seed)
    rows, golds, rerolls = [], [], 0
    for i, (k, cls, pc) in enumerate(specs):
        for _try in range(2000):
            res = build_task(rng, k, cls, pc)
            if res is not None:
                row, ref = res
                rows.append(row)
                golds.append(ref)
                break
            rerolls += 1
        else:
            raise RuntimeError(f"could not build task {i} spec={(k, cls, pc)} after 2000 tries")
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_gold, 'w') as f:
        for g in golds:
            f.write(json.dumps(g) + "\n")
    n_plan = sum(1 for g in golds if g and g.get('is_feasible'))
    n_ref = sum(1 for g in golds if g and not g.get('is_feasible'))
    print(f"[generate] {len(rows)} rows -> {out_csv} ; gold = {n_plan} schedule-complete plans "
          f"+ {n_ref} refusals -> {out_gold} ; rerolls={rerolls}")
    return rows


# ======================================================================================
# §3  SCHEDULE-COMPLETE GOLD PLANS
#
# The cost bundle (§1.5) proves a task is affordable; it cannot exercise B2 (opening hours)
# or B3 (spatio-temporal feasibility) because it carries no times. This section turns the
# same argmin picks into a day-by-day plan in the agent's own output schema, so the gold can
# be scored on every dimension.
# ======================================================================================
SIGHT_SLOTS = [10 * 60, 14 * 60]     # two generous, well-separated sightseeing slots
CHECKIN_LAG_MIN = 90                 # land -> hotel check-in
HOTEL_CHECKIN_EARLIEST = 14 * 60     # standard hotel check-in hour
MAX_PER_DAY = 3                      # a full day holds at most three visits (measured: 3 x <=3h
                                     # with 60-min gaps ends by 18:00, and 100% of attractions are
                                     # open until 17:00; four never fits)


def _hm(mins):
    mins = max(0, min(23 * 60 + 59, int(mins)))
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _mins(hm):
    try:
        h, m = str(hm).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _parse_open(oh):
    try:
        a, b = str(oh).split("-")
        return _mins(a), _mins(b)
    except Exception:
        return None, None


def _duration_min(s, default=120):
    import re as _re
    mt = _re.search(r"(\d+(?:\.\d+)?)", str(s))
    if not mt:
        return default
    v = float(mt.group(1))
    return int(v * 60) if "hour" in str(s).lower() else int(v)


MAX_VISIT_MIN = 180        # the gold spends at most 3h at any one attraction
MIN_VISIT_FRAC = 0.6       # refuse a slot that cannot host at least this much of the intended visit
FALLBACK_GAP_MIN = 60      # used only when a coordinate is unavailable
_COORD_CACHE = {}


def _entity_coord(kind, city, name):
    """(lat, lon) of a KB hotel or attraction, or None."""
    key = (kind, city, name)
    if key in _COORD_CACHE:
        return _COORD_CACHE[key]
    df = load_attr(city) if kind == "attraction" else load_hotels(city)
    col = "attraction_name" if kind == "attraction" else "name"
    out = None
    if df is not None:
        m = df[df[col].astype(str) == str(name)]
        if not m.empty:
            r = m.iloc[0]
            out = (float(r["latitude"]), float(r["longitude"]))
    _COORD_CACHE[key] = out
    return out


def _persona_day_window(personas):
    """(day_start, latest_end) for a sightseeing day. The slot grid used to be a module constant
    consulted without ever looking at the personas: 25 of 73 nightlife itineraries had no visit
    ending after 18:00, while a with-children plan could run a nightclub to 21:00."""
    base, latest = SIGHT_SLOTS[0], None
    if "nightlife enthusiast" in personas:
        base = 13 * 60                       # start later so the evening is actually reachable
    if "with children" in personas or "elderly travelers" in personas:
        latest = 20 * 60                     # nothing running past 20:00
    return base, latest


def _hav_xy(a, b):
    """Great-circle km between two (lat, lon) points."""
    from scoring import haversine_distance
    if not a or not b:
        return 0.0
    return haversine_distance(a[0], a[1], b[0], b[1])


def _travel_min(a, b):
    """Minutes the SCORER will require between two (lat, lon) points — deliberately the very
    function B3 uses, so the gold can never schedule a hop the scorer then calls infeasible.
    Replaces the old flat 60-minute inter-visit gap, which assumed 300 km/h ground transport."""
    from scoring import get_min_travel_time, haversine_distance
    if not a or not b:
        return FALLBACK_GAP_MIN
    return get_min_travel_time(haversine_distance(a[0], a[1], b[0], b[1]))


def _attr_entry(city, name, earliest, cap=MAX_VISIT_MIN):
    """One attraction visit starting no earlier than `earliest`, clamped so visit_start always falls
    inside open_hours (B2). Returns (entry, start, ok); `ok` is False when the remaining window is
    too short to be a real visit — those were being emitted as 30-minute stubs at closing time
    (154 of them, every one starting at exactly close-minus-30, full ticket billed)."""
    A = load_attr(city)
    row = A[A["attraction_name"] == name].iloc[0]
    o, cl = _parse_open(row["open_hours"])
    want = min(_duration_min(row.get("duration_of_visit")), cap)
    if o is None or cl is None:
        start, end = earliest, earliest + want
    else:
        start = min(max(earliest, o), max(o, cl - 30))
        end = min(start + want, cl)
    ok = (start >= earliest) and ((end - start) >= MIN_VISIT_FRAC * want)
    return {"name": name, "city": city, "visit_start": _hm(start), "visit_end": _hm(end)}, start, ok


def build_schedule(meta, personas, req, R):
    """Day-by-day, schedule-complete gold plan built from the SAME memoized argmin picks the cost
    model uses.

    Every gap is sized with `_travel_min`, i.e. the scorer's own travel-time function over real KB
    coordinates, so the gold can never contain a hop B3 would call infeasible. Attractions are
    spread ROUND-ROBIN across a city's free days (front-loading used to leave 72 stay-days empty),
    a transfer day earns a morning slot in the ORIGIN city bounded by the outbound flight (all 191
    transfer days with a >=15:00 departure used to be empty), an overnight leg advances the day
    index so check-in lands on the day the traveller physically enters the room, and a one-way
    trip's trailing day gets no sightseeing because it has neither a room nor a flight out.
    """
    cities, idx, nights, cardays, _bn, _bd = city_maps(meta)
    if not cities:
        return None
    dep = (meta.req_flight or {}).get("departure_city")
    round_trip = (meta.req_flight or {}).get("trip_type") == "round"

    H, C = {}, {}
    for c in cities:
        ph = pick_hotel(c, personas, req.get("hotel", {}))
        if ph is None:
            return None
        H[c] = ph
        if "car" in R:
            pc = pick_car(c, meta.person_num, personas, req.get("car", {}))
            if pc is None:
                return None
            C[c] = pc

    final_fi = flight_info(cities[-1], dep, civil=True) if round_trip else None
    if round_trip and final_fi is None:
        return None

    def _city_day(c):
        d = {"current_city": c, "flights": [], "attractions": [],
             "hotel": {"name": H[c][1], "city": c, "price_per_night": H[c][0]}}
        if "car" in R:
            d["car"] = {"car_type": C[c][1], "city": c, "price_per_day": C[c][0]}
        return d

    def _leg(fi, a, b):
        return {"flight_number": fi["flight_number"], "departure_city": a, "arrival_city": b,
                "departure_time": fi["departure_time"], "arrival_time": fi["arrival_time"],
                "price": fi["price"]}

    # slot = (day_key, city, base_time, anchor_coord, deadline_time, deadline_coord, cap)
    plan, day, prev, slots = {}, 1, dep, {c: [] for c in cities}

    for i, c in enumerate(cities):
        ad = day
        fi = flight_info(prev, c, civil=True)
        if fi is None:
            return None
        dm, arr = _mins(fi["departure_time"]), _mins(fi["arrival_time"])
        overnight = dm is not None and arr is not None and arr < dm
        hotel_xy = _entity_coord("hotel", c, H[c][1])

        d = _city_day(c)
        d["current_city"] = f"{prev} to {c}"
        d["flights"] = [_leg(fi, prev, c)]
        lag = max(CHECKIN_LAG_MIN, _travel_min(fi.get("arr_coord"), hotel_xy))
        # no front desk hands a room over at 09:15 -- floor the check-in at the standard hour.
        # This only lifts EARLY check-ins; a late arrival keeps its (possibly next-day) time.
        ci = max((arr if arr is not None else 12 * 60) + lag, HOTEL_CHECKIN_EARLIEST)
        rolls_over = overnight or ci > 23 * 60 + 59
        if not rolls_over:
            d["hotel"]["check_in"] = _hm(ci)
        plan[f"day{ad}"] = d
        for off in range(1, nights[i]):
            plan[f"day{ad + off}"] = _city_day(c)
        rolled = None
        if rolls_over and nights[i] >= 2:
            rolled = ci % (24 * 60)
            plan[f"day{ad + 1}"]["hotel"]["check_in"] = _hm(rolled)

        # morning sightseeing in the ORIGIN city before this flight leaves (same rule the
        # fly-home day already used; 143 fly-home days prove it works)
        if i > 0 and dm is not None:
            o = cities[i - 1]
            # NB: the scorer locates a flight event at its ARRIVAL airport (data_loader
            # get_coordinates, entity_type == "flight"), so the gap it will measure from a
            # morning visit is visit -> DESTINATION airport, not visit -> local airport.
            slots[o].append((f"day{ad}", o, SIGHT_SLOTS[0], _entity_coord("hotel", o, H[o][1]),
                             dm, fi.get("arr_coord"), 1))
        # arrival-day sightseeing only when the traveller actually checked in today
        _dstart, _dlatest = _persona_day_window(personas)
        if not rolls_over:
            base = max(ci, dm if dm is not None else 0)
            if base <= 15 * 60:
                slots[c].append((f"day{ad}", c, base, hotel_xy, _dlatest, None, 1))
        for off in range(1, nights[i]):
            base = _dstart
            if rolled is not None and off == 1:
                # the traveller only enters the room at `rolled` on this day, and the scorer sees a
                # check-in event there -- sightseeing cannot precede it
                base = max(base, rolled)
            slots[c].append((f"day{ad + off}", c, base, hotel_xy, _dlatest, None, MAX_PER_DAY))
        day = ad + nights[i]
        prev = c

    # final day: a CHECKOUT day, so no hotel either way; the car is billed for all `days`
    last = cities[-1]
    final = {"current_city": last, "flights": [], "attractions": []}
    if "car" in R:
        final["car"] = {"car_type": C[last][1], "city": last, "price_per_day": C[last][0]}
    if round_trip:
        final["current_city"] = f"{last} to {dep}"
        final["flights"] = [_leg(final_fi, last, dep)]
        fdm = _mins(final_fi["departure_time"])
        if fdm is not None and fdm - 120 >= 12 * 60:
            slots[last].append((f"day{day}", last, SIGHT_SLOTS[0],
                                _entity_coord("hotel", last, H[last][1]),
                                fdm, final_fi.get("arr_coord"), 1))
    # one-way: the trailing day has neither a room nor a flight out, so it gets NO sightseeing
    plan[f"day{day}"] = final

    # fill: round-robin over each city's slots, earlier-opening attractions first, named ones first
    req_named = set((req.get("attraction") or {}).get("name") or [])
    for c in cities:
        cap_total = sum(s[6] for s in slots[c])
        picks = pick_tickets(c, personas, max(A_SCHED, cap_total), allow_fewer=True)
        if not picks:
            return None
        ordered = sorted((nm for _p, nm in picks), key=lambda nm: _open_key(c, nm))
        pending = ([nm for nm in ordered if nm in req_named]
                   + [nm for nm in ordered if nm not in req_named])
        cursor_t = {s[0]: s[2] for s in slots[c]}
        cursor_xy = {s[0]: s[3] for s in slots[c]}
        placed = {s[0]: 0 for s in slots[c]}
        for _round in range(MAX_PER_DAY):
            for dk, scity, _b, _a, dl_t, dl_xy, cap in slots[c]:
                if not pending or placed[dk] >= cap:
                    continue
                j = 0
                while j < len(pending):
                    nm = pending[j]
                    xy = _entity_coord("attraction", scity, nm)
                    earliest = cursor_t[dk] + _travel_min(cursor_xy[dk], xy)
                    entry, start, ok = _attr_entry(scity, nm, earliest)
                    end = _mins(entry["visit_end"])
                    _need = _travel_min(xy, dl_xy) if dl_xy is not None else 0
                    if not ok or (dl_t is not None and end + _need > dl_t):
                        j += 1
                        continue
                    plan[dk]["attractions"].append(entry)
                    pending.pop(j)
                    cursor_t[dk], cursor_xy[dk] = end, xy
                    placed[dk] += 1
                    break
    # D-5b: the cost-optimal SET is fixed, but its ORDER within a day is free. Re-lay each day in
    # every permutation and keep the shortest ground path that is still fully feasible, so B2/B3
    # can never regress. <=3 stops per day, so this is exhaustive and instant.
    from itertools import permutations
    for c in cities:
        for dk, scity, base, anchor, dl_t, dl_xy, _cap in slots[c]:
            day = plan.get(dk)
            if not day or len(day.get("attractions", [])) < 2:
                continue
            names = [a["name"] for a in day["attractions"]]
            best, best_d = None, None
            for perm in permutations(names):
                laid, dist = _lay_day(scity, perm, base, anchor, dl_t, dl_xy)
                if laid is not None and (best_d is None or dist < best_d - 1e-9):
                    best, best_d = laid, dist
            if best:
                day["attractions"] = best
    for dk in plan:
        plan[dk]["attractions"].sort(key=lambda a: _mins(a["visit_start"]) or 0)
    return {"is_feasible": True, "refusal_reason": "", "plan": plan}


def _lay_day(scity, names, base, anchor, dl_t, dl_xy):
    """Re-time `names` in this order from (base, anchor). Returns (entries, ground_km) or
    (None, None) if the order is not feasible."""
    out, t, xy, dist = [], base, anchor, 0.0
    for nm in names:
        axy = _entity_coord("attraction", scity, nm)
        entry, start, ok = _attr_entry(scity, nm, t + _travel_min(xy, axy))
        end = _mins(entry["visit_end"])
        _need = _travel_min(axy, dl_xy) if dl_xy is not None else 0
        if not ok or (dl_t is not None and end + _need > dl_t):
            return None, None
        out.append(entry)
        dist += _hav_xy(xy, axy)
        t, xy = end, axy
    return out, dist


# --------------------------------------------------------------------------------------
# §1 smoke test
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import types

    _build_routes()
    G = route_graph()
    print(f"[routes] {len(G)} directed edges")
    print(f"[Paris->Amsterdam] {cheapest_flight('Paris', 'Amsterdam')}")
    print(f"[Amsterdam->Paris] {cheapest_flight('Amsterdam', 'Paris')}")

    usable = [c for c in {a for a, _ in G} | {b for _, b in G} if has_all_domains(c)]
    print(f"[usable cities] {len(usable)} have flights + all 3 KB domains")

    # single-city solvable-style meta: Paris, round trip, 2 people, 1 persona
    def M(dep, arr_list, days, person, trip="round"):
        m = types.SimpleNamespace(
            req_flight={"departure_city": dep, "arrival_city": arr_list, "trip_type": trip},
            days=days, person_num=person, rooms_count=rooms_for(person), cities_count=len(arr_list))
        return m

    if cheapest_flight("Paris", "Amsterdam"):
        m1 = M("Paris", ["Amsterdam"], 3, 2)
        req = {"hotel": {}, "car": {}, "attraction": {"name": []}}
        for personas in ([], ["luxury travelers"], ["foodie"], ["business travelers", "photography"]):
            cf = C_floor(m1, req, {"flight", "hotel", "attraction"})
            cm = C_min_full(m1, personas, req, {"flight", "hotel", "attraction"})
            ok = (cf is not None and cm is not None and cf <= cm)
            print(f"[Paris->Amsterdam 3d/2p personas={personas}] C_floor={cf} C_min_full={cm} floor<=min:{ok}")

    # multi-city: pick a connected 3-clique from usable
    print("[multi-city probe]")
    found = 0
    ul = sorted(usable)
    for a in ul[:60]:
        outs = [b for b in ul if (a, b) in G and (b, a) in G]
        if len(outs) >= 2:
            b, c = outs[0], outs[1]
            if (b, c) in G or (c, b) in G:
                m3 = M(a, [b, c], 9, 3)
                req = {"hotel": {}, "car": {}, "attraction": {"name": []}}
                cf = C_floor(m3, req, {"flight", "hotel", "attraction"})
                cm = C_min_full(m3, ["business travelers"], req, {"flight", "hotel", "attraction", "car"})
                print(f"  {a}->{b}->{c} 9d/3p: C_floor={cf} C_min_full(+car,business)={cm}")
                found += 1
                if found >= 2:
                    break

    # --- self-verify: scorer's bill of the reference plan == C_min_full (correct-by-construction) ---
    print("[self-verify vs real scorer]")
    req = {"hotel": {}, "car": {}, "attraction": {"name": []}}
    checks = [
        (M("Paris", ["Amsterdam"], 3, 2), ["luxury travelers"], {"flight", "hotel", "attraction"}),
        (M("Paris", ["Amsterdam"], 3, 2), ["business travelers"], {"flight", "hotel", "attraction", "car"}),
    ]
    for m, personas, R in checks:
        cm = C_min_full(m, personas, req, R)
        plan = build_reference_plan(m, personas, req, R, feasible=True)
        billed = scorer_cost(plan, m) if plan else None
        match = (cm is not None and billed is not None and abs(cm - billed) < 1e-6)
        print(f"  {m.req_flight['arrival_city']} personas={personas} R={sorted(R)}: "
              f"C_min_full={cm} scorer_bill={billed} MATCH={match}")
        assert match, "correct-by-construction VIOLATED: C_min_full != scorer bill"
    print("[smoke test done]")
