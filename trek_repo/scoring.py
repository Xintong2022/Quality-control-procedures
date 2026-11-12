"""
KDD Travel Planning - Scoring Script
=====================================
多维度评分系统，评估 LLM 生成的旅行计划质量。

评分维度:
- D0: Explicit (D0-key: 关键词命中, D0-src: 数据一致性)
- D1: Implicit (隐式需求命中)
- D2: City Dimension (4列: Ordered/Unordered × Single/Multi)
- D3: Budget Trade-off (预算指数衰减)
- D4: Impossible Handling (不可行任务处理)
- D5: Retry Robustness (重试稳健性)
- CCR: B1 (景点超量), B2 (营业时间合规), B3 (时空可行性)
"""

import os
import sys
import json
import math
import argparse
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime

import pandas as pd

from data_loader import QueryMeta, load_queries, SandboxDB, get_sandbox_db
from implicit_scoring import score_d1_implicit_v2, get_implicit_scorer
from cost_model import city_maps, _fold  # shared per-city night/car-day allocation


# ============ 评分结果数据结构 ============
@dataclass
class ScoreResult:
    """单个查询的评分结果"""
    query_index: int
    
    # D0: Explicit
    d0_keyword: Optional[float] = None
    d0_source: Optional[float] = None
    
    # D1: Implicit
    d1_implicit: Optional[float] = None
    
    # D2: City Dimension (2 columns)
    d2_unord_single: Optional[float] = None
    d2_unord_multi: Optional[float] = None
    
    # D3: Budget Trade-off
    d3_budget: Optional[float] = None
    
    # D4: Impossible Handling
    d4_impossible: Optional[float] = None
    
    # D5: Retry Robustness
    d5_retry: Optional[float] = None
    
    # CCR
    b2_opening_hours: Optional[float] = None
    b3_spatiotemporal: Optional[float] = None
    
    # 辅助信息
    # Severity diagnostic behind the binary d0_source: the FRACTION of named entities that verified.
    # d0_source itself is 0/1 (any fabrication zeroes the task); this keeps "one bad name" and
    # "everything invented" distinguishable when reporting.
    d0_source_ratio: Optional[float] = None
    # Hard-gate outcome: did this itinerary pass every validity gate (delivered substance, invented
    # nothing, honoured every explicitly mandated entity)? Reported as the benchmark's headline
    # binary number — the "fully valid plan rate" — alongside the graded quality dimensions.
    fully_valid: Optional[bool] = None
    # Whether the TASK is labelled impossible — set unconditionally in score_query so the
    # aggregation can scope cat_infeasibility to the infeasible third (see compute_aggregate_scores).
    task_impossible: Optional[bool] = None
    # Task-derived per-task success: gates pass on a feasible task, correct refusal on an
    # infeasible one. Always set, so its denominator is the whole benchmark for every model.
    task_success: Optional[bool] = None
    invalid_reasons: Optional[list] = None
    # Cost of producing the answer (from the run record, not the plan).
    total_tokens: Optional[int] = None
    elapsed_sec: Optional[float] = None
    total_cost: Optional[float] = None
    attraction_count: int = 0
    is_feasible: bool = False


# ============ 辅助函数 ============
def parse_time(time_str: str) -> Optional[int]:
    """解析 HH:MM 格式时间为分钟数"""
    if not time_str or time_str == "-":
        return None
    try:
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return None


def required_legs(meta) -> list:
    """The direct flight legs the task requires, in itinerary order.

    Every task states 'direct flights only', so the itinerary is exactly
    departure -> city1 -> ... -> cityN (-> departure for a round trip).
    """
    rf = getattr(meta, "req_flight", None) or {}
    dep = rf.get("departure_city")
    arr = rf.get("arrival_city") or []
    if isinstance(arr, str):
        arr = [arr]
    chain = ([dep] + [c for c in arr if c]) if dep else [c for c in arr if c]
    if str(rf.get("trip_type", "")).lower().startswith("round") and dep and len(chain) > 1:
        chain = chain + [dep]
    return [(a, b) for a, b in zip(chain, chain[1:]) if a and b]


def required_entity_slots(meta) -> int:
    """How many KB-grounded entities a compliant itinerary must name.

    One per required flight leg, one hotel per stay city, one car per stay city when the task asks
    for a car. Attractions are deliberately excluded — the task never fixes a visit COUNT, so
    sightseeing coverage is measured separately rather than smuggled into the grounding denominator.

    This is what stops D0-src (the entire 25% Truthfulness category) from being a pure ratio that a
    plan naming exactly ONE real hotel could max at 1.000, tying the gold itinerary.
    """
    from cost_model import stay_cities_in_order
    cities = stay_cities_in_order(meta) or []
    slots = len(required_legs(meta))
    if getattr(meta, "req_hotel", None):
        slots += max(1, len(cities))
    if getattr(meta, "req_car", None):
        slots += max(1, len(cities))
    return slots


_CLASS_PATTERNS = {
    "entity": (r"\bdoes not exist\b|\bdoesn't exist\b|\bnot (?:in|found in) the (?:knowledge base|database|kb)\b"
               r"|\bno such (?:hotel|attraction|car|place)\b|\bnot available in the (?:kb|database)\b"
               r"|\bfictional\b|\bcould not (?:be )?find\b.*\b(?:hotel|attraction)\b"),
    "route": (r"\bno (?:direct )?flight\b|\bno route\b|\bnot connected\b|\bno .*connection\b"
              r"|\bdirect flight\b.*\b(?:does not|doesn't|no)\b|\bunreachable\b|\bno way to fly\b"),
    "budget": (r"\bbudget\b|\btoo expensive\b|\bcannot afford\b|\bcan't afford\b|\bexceeds? the\b.*\bcost\b"
               r"|\binsufficient funds\b|\bbelow the (?:minimum|cheapest)\b"),
}


def stated_impossibility_class(reason: str):
    """Classify the agent's refusal text into entity / route / budget, or None if it says nothing.

    Checked most-specific first: a reason may mention 'budget' incidentally while the real finding
    is a missing entity or leg.
    """
    if not reason:
        return None
    txt = str(reason).lower()
    for cls in ("entity", "route", "budget"):
        if re.search(_CLASS_PATTERNS[cls], txt):
            return cls
    return None


def true_impossibility_class(meta, db):
    """Independently re-derive WHY a task is infeasible, from the task + the KB.

    Deliberately does not trust a stored label: the scorer establishes the ground truth itself, in
    the same order the generator constructs it (a named entity that is absent, then a required
    direct leg that does not exist, then a budget under the achievable floor).

    Without this, the 267 infeasible tasks — a third of the benchmark — were graded on a single
    boolean, and 'I refuse' scored the same as correctly diagnosing the cause.
    """
    # 1. a specifically named entity that is not in the KB
    from cost_model import stay_cities_in_order
    cities = stay_cities_in_order(meta) or []
    named = []
    if getattr(meta, "req_hotel", None) and meta.req_hotel.get("name"):
        named.append(("hotel", meta.req_hotel["name"]))
    ra = getattr(meta, "req_attraction", None) or {}
    for nm in (ra.get("name") or []):
        named.append(("attraction", nm))
    if getattr(meta, "req_car", None) and meta.req_car.get("car_type"):
        named.append(("car", meta.req_car["car_type"]))
    for kind, nm in named:
        present = False
        for c in cities:
            if kind == "hotel" and db.verify_hotel(nm, c):
                present = True
            elif kind == "attraction" and db.verify_attraction(nm, c):
                present = True
            elif kind == "car" and db.verify_car(nm, c):
                present = True
            if present:
                break
        if not present:
            return "entity"
    # 2. a required direct leg the KB does not serve
    for a, b in required_legs(meta):
        if db.cheapest_flight_price(a, b) is None:
            return "route"
    # 3. otherwise it is the budget
    return "budget"


def mandated_entities(meta) -> list:
    """Entities the query explicitly REQUIRES ("We must book X only, nothing else will do")."""
    out = []
    rh = getattr(meta, "req_hotel", None) or {}
    if rh.get("name"):
        out.append(("hotel", rh["name"]))
    ra = getattr(meta, "req_attraction", None) or {}
    for nm in (ra.get("name") or []):
        out.append(("attraction", nm))
    rc = getattr(meta, "req_car", None) or {}
    if rc.get("car_type"):
        out.append(("car", rc["car_type"]))
    return out


def _day_order(key):
    """Sort day keys by their NUMBER, not lexicographically.

    Plain `sorted()` orders day10 before day2, which would misalign per-night hotel rates with the
    nights they belong to on any trip longer than nine days.
    """
    m = re.search(r"(\d+)", str(key))
    return (0, int(m.group(1))) if m else (1, str(key))


def plan_validity_gates(plan_data: dict, meta, db) -> tuple:
    """HARD GATES — binary per itinerary. Returns (passed, reasons).

    A gate is a failure that makes the itinerary unusable or dishonest, where partial compliance
    carries no partial value:
      * delivered nothing                      — there is no itinerary to execute
      * invented an entity                     — the traveller arrives to find it does not exist
      * ignored an explicitly mandated entity  — the user said "we must book X, nothing else will do"
    Graded quality (scheduling, persona fit, city coverage) is scored separately: those DO have
    partial value, so binarising them would only destroy discrimination.
    """
    reasons = []
    if not _plan_books_anything(plan_data):
        reasons.append("empty_plan")

    # MAJORITY OF VISITS UN-ENTERABLE. An itinerary whose scheduled visits mostly fall outside the
    # attractions' opening hours cannot be walked: the traveller stands at locked doors. B2 grades
    # this proportionally, but a plan can be wrong on B2 while every OTHER category stays high — the
    # audit measured otherwise-perfect itineraries with b2=0 keeping a per-task geomean of 0.837
    # because the misscheduling only ever touches 1/5 of the aggregate. Partial compliance below
    # half carries no partial value (a trip where most doors are locked is not partially usable), so
    # this is a gate, not a grade. Fires on 0 of the 800 gold plans (gold's feasible-task b2 floor
    # is 1.0). Requires >= 2 checkable visits so a single mistimed visit stays B2's business.
    _checkable = _outside = 0
    for _a in plan_data.get("attractions", []) or []:
        _nm, _cy = _a.get("name"), _a.get("city")
        _st, _en = parse_time(_a.get("visit_start")), parse_time(_a.get("visit_end"))
        if not _nm or not _cy or _st is None or _en is None:
            continue
        _oh = db.get_open_hours(_nm, _cy)
        if not _oh or _oh == "-":
            continue
        _om, _cm = _parse_hours_range(_oh)
        if _om is None or _cm is None:
            continue
        _checkable += 1
        if _st < _om or _en > _cm:
            _outside += 1
    if _checkable >= 2 and _outside * 2 > _checkable:
        reasons.append(f"majority_visits_outside_hours:{_outside}/{_checkable}")
    for f in plan_data.get("flights", []) or []:
        fn = f.get("flight_number")
        if fn and not db.verify_flight(fn):
            reasons.append(f"fabricated_flight:{fn}")
            break
    for h in plan_data.get("hotels", []) or []:
        nm, c = h.get("name"), h.get("city")
        if nm and (not c or not db.verify_hotel(nm, c)):
            reasons.append(f"fabricated_hotel:{nm}")
            break
    for a in plan_data.get("attractions", []) or []:
        nm, c = a.get("name"), a.get("city")
        if nm and (not c or not db.verify_attraction(nm, c)):
            reasons.append(f"fabricated_attraction:{nm}")
            break
    # Cars were the one entity type these gates skipped, so an itinerary booking a "Hovercraft"
    # stayed fully_valid — and therefore counted as a SUCCESSFUL task — even though D0-src had
    # already scored it 0 for the same fabrication. The gates and the truthfulness dimension must
    # not disagree about whether a submission invented something.
    for c_ in plan_data.get("cars", []) or []:
        ct, c = (c_.get("car_type") or c_.get("type")), c_.get("city")
        if ct and (not c or not db.verify_car(ct, c)):
            reasons.append(f"fabricated_car:{ct}")
            break
    # explicitly mandated entities must actually appear in the plan
    for kind, nm in mandated_entities(meta):
        key = str(nm).strip().lower()
        if kind == "hotel":
            got = any(str(h.get("name", "")).strip().lower() == key for h in plan_data.get("hotels", []) or [])
        elif kind == "attraction":
            got = any(str(a.get("name", "")).strip().lower() == key for a in plan_data.get("attractions", []) or [])
        else:
            got = any(str(c.get("car_type") or c.get("type") or "").strip().lower() == key
                      for c in plan_data.get("cars", []) or [])
        if not got:
            reasons.append(f"missing_mandated_{kind}:{nm}")
    return (not reasons), reasons


def _plan_books_anything(plan_data: dict) -> bool:
    """True if the submission actually books a resource of any kind.

    Used to separate "delivered an itinerary" from "claimed feasibility and delivered nothing".
    Several dimensions (D3, D4, D5) must not award credit to the latter — an empty plan used to
    score D3 = 1.0, D4 = 1.0 and D5 = 0.88, i.e. roughly half the benchmark, for submitting `{}`.
    """
    return bool(plan_data.get("flights") or plan_data.get("hotels")
                or plan_data.get("attractions") or plan_data.get("cars"))


def _ci_get(d: dict, *keys):
    """Case-insensitive dict lookup over field aliases.

    extract_plan_data is alias/case-insensitive but B3 read raw lowercase keys, so a plan that
    wrote `Flights` or `Attractions` produced no events for B3 and deleted the dimension. Every
    reader of a submitted plan must use the same tolerant rule.
    """
    if not isinstance(d, dict):
        return None
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if str(k).lower() in low:
            return low[str(k).lower()]
    return None


def _parse_hours_range(open_hours: str):
    """('09:00-17:30') -> (540, 1050) in minutes, or (None, None) if unparseable.

    Unlike `is_time_in_range`, this does NOT fail open: an unparseable range yields None so the
    caller can treat the visit as unverifiable rather than silently compliant.
    """
    if not open_hours or open_hours == "-":
        return None, None
    parts = str(open_hours).split("-")
    if len(parts) != 2:
        return None, None
    a, b = parse_time(parts[0].strip()), parse_time(parts[1].strip())
    if a is None or b is None:
        return None, None
    return a, b


def is_time_in_range(visit_time: str, open_hours: str) -> bool:
    """检查访问时间是否在营业时间内"""
    if not open_hours or open_hours == "-":
        return True  # 无营业时间数据，默认通过
    
    visit_min = parse_time(visit_time)
    if visit_min is None:
        return True
    
    # 解析营业时间 (格式: "09:00-17:30")
    try:
        parts = open_hours.split("-")
        if len(parts) != 2:
            return True
        open_min = parse_time(parts[0].strip())
        close_min = parse_time(parts[1].strip())
        if open_min is None or close_min is None:
            return True
        return open_min <= visit_min <= close_min
    except:
        return True


# The B3 travel model lives in travel_time.py — the SINGLE source shared with the agent-facing
# `compute_travel_time` tool, so the gap B3 requires equals the gap the tool reports to the agent.
from travel_time import haversine_km as _haversine_km, min_travel_minutes as _min_travel_minutes


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的 Haversine 距离 (km)。见 travel_time.py。"""
    return _haversine_km(lat1, lon1, lat2, lon2)


def get_min_travel_time(distance_km: float) -> int:
    """根据距离返回最小旅行时间 (分钟)。见 travel_time.py (agent 的 compute_travel_time 工具同源)。"""
    return _min_travel_minutes(distance_km)


# Token budget for D5, calibrated on measured runs (~6k tokens per tool call under the lossless
# notebook, plus a fixed system-prompt/overhead term). Absolute and task-derived so the metric is
# reproducible across different model line-ups.
# Derived from first principles, NOT fitted to one model. The lossless notebook resends the whole
# transcript every turn, so context grows QUADRATICALLY in the number of tool calls: turn t carries
# the results of all t-1 previous calls. Over T = min_calls turns the total is
#     base*T + per_result * T(T+1)/2
# At T=8 this is ~124k, which matches the measured median across 11 models (101k). The first
# calibration (a flat 8k + 6k*T = 54k) was fitted to the ONE model with the lowest token use in a
# 3-query smoke test, and consequently scored 9 of 11 models as 40-436% over budget — it measured
# "did you use fewer tokens than Grok", not efficiency.
TOKEN_BUDGET_BASE = 2000        # per-turn fixed overhead (system prompt, query, plan scaffolding)
TOKEN_BUDGET_PER_CALL = 3000    # tokens one search result contributes, re-sent on every later turn
TOKEN_OVERRUN_BETA = 1.0        # 2x the budget -> exp(-1) = 0.37


# ============ 评分函数 ============
class TravelPlanScorer:
    """旅行计划评分器"""
    
    def __init__(self, db: SandboxDB, config: dict = None):
        self.db = db
        self.config = config or {}
        
        # 可配置参数
        self.beta = self.config.get("beta", 4.0)  # D3 预算衰减系数
    
    def extract_plan_data(self, plan_output: dict) -> dict:
        """从 plan_output 提取结构化数据。

        Case-insensitive on container keys (`Flights`/`flight`), reads every field alias
        (`name`/`hotel_name`, `type`/`car_type`), keeps STRING-form flights so they are verified
        rather than silently dropped, and dedups all four entity types. The old extractor read
        hotels only via `name`, hit a dead `pass` on string flights, and skipped city-less
        attractions — so a fabrication placed in any of those escaped D0-src truthfulness entirely.
        """
        def _ci(d, *keys):
            if not isinstance(d, dict):
                return None
            low = {str(k).lower(): v for k, v in d.items()}
            for k in keys:
                if str(k).lower() in low:
                    return low[str(k).lower()]
            return None

        is_feasible = _ci(plan_output, "is_feasible")
        is_feasible = bool(is_feasible) if is_feasible is not None else False
        plan = _ci(plan_output, "plan") or {}
        if not isinstance(plan, dict):
            plan = {}

        flights, hotels, cars, attractions, cities = [], [], [], [], []
        f_seen, h_seen, c_seen, a_seen = set(), set(), set(), set()

        for day_key in sorted(plan.keys(), key=str):
            day = plan.get(day_key)
            if not isinstance(day, dict):
                continue

            dfl = _ci(day, "flights", "flight")
            if isinstance(dfl, dict):
                dfl = [dfl]
            if isinstance(dfl, list):
                for f in dfl:
                    if isinstance(f, dict):
                        fn = _ci(f, "flight_number", "flightnumber")
                        if fn and str(fn) != "-":
                            k = str(fn).strip().lower()
                            if k not in f_seen:
                                f_seen.add(k)
                                flights.append(f)
                    elif isinstance(f, str) and f.strip() and f.strip() != "-":
                        k = f.strip().lower()
                        if k not in f_seen:
                            f_seen.add(k)
                            flights.append({"flight_number": f.strip()})

            hotel = _ci(day, "hotel", "hotels")
            if isinstance(hotel, list):
                hotel = hotel[0] if hotel else None
            if isinstance(hotel, dict):
                name = _ci(hotel, "name", "hotel_name")
                if name and str(name) != "-":
                    city = _ci(hotel, "city") or ""
                    k = (str(name).lower(), str(city).lower())
                    if k not in h_seen:
                        h_seen.add(k)
                        hotels.append({**hotel, "name": name})

            car = _ci(day, "car", "cars")
            if isinstance(car, list):
                car = car[0] if car else None
            if isinstance(car, dict):
                ctype = _ci(car, "type", "car_type")
                if ctype and str(ctype) != "-":
                    city = _ci(car, "city") or ""
                    k = (str(ctype).lower(), str(city).lower())
                    if k not in c_seen:
                        c_seen.add(k)
                        cars.append({**car, "car_type": ctype, "type": ctype})

            datt = _ci(day, "attractions", "attraction")
            if isinstance(datt, dict):
                datt = [datt]
            if isinstance(datt, list):
                for a in datt:
                    if not isinstance(a, dict):
                        continue
                    name = _ci(a, "name", "attraction_name")
                    if name and str(name) != "-":
                        city = _ci(a, "city") or ""
                        k = (str(name).lower(), str(city).lower())
                        if k not in a_seen:
                            a_seen.add(k)
                            attractions.append({**a, "name": name})

            cur = _ci(day, "current_city", "currentcity") or ""
            if isinstance(cur, str) and cur:
                for part in re.split(r"\s+to\s+|\s*->\s*", cur, flags=re.IGNORECASE):
                    part = part.strip()
                    if part.lower().startswith("from "):
                        part = part[5:].strip()
                    if part and part not in cities:
                        cities.append(part)

        return {
            "is_feasible": is_feasible,
            "refusal_reason": _ci(plan_output, "refusal_reason", "reason") or "",
            "flights": flights,
            "attractions": attractions,
            "hotels": hotels,
            "cars": cars,
            "cities": cities,
            "plan": plan,
        }

    def score_d0_keyword(self, plan_data: dict, meta: QueryMeta) -> float:
        """D0-1: 显式关键词命中率（严格字段匹配）"""
        checks = []  # 所有需要检查的约束
        matched = 0  # 命中数量
        
        # ---- Flights are checked LEG BY LEG, not city by city. ----
        # Checking only "some flight departs from D" and "some flight reaches each arrival city"
        # never verified the itinerary actually closes: on the 373 round-trip feasible tasks,
        # deleting the return leg was score-identical. Every task states direct-flights-only, so the
        # required legs are exactly D->c1->...->cN(->D), and each is its own constraint.
        booked_legs = {(_fold(str(f.get("departure_city") or "")), _fold(str(f.get("arrival_city") or "")))
                       for f in plan_data["flights"]}
        for a, b in required_legs(meta):
            checks.append(("flight_leg", f"{a}->{b}"))
            if (_fold(a), _fold(b)) in booked_legs:
                matched += 1

        # ---- Itinerary completeness: every day present, AND lodging in every stay city. ----
        # Nothing previously required the plan to span the requested duration, so a 9-day trip
        # delivered as a single day key lost nothing.
        # A day counts toward coverage only if it carries something. `isinstance(v, dict)` alone let
        # a 9-day trip crammed into day1 buy back the coverage check with eight `{}` placeholders
        # (D0-key 0.727 -> 0.818): the check exists to make the itinerary actually span the requested
        # duration, and an empty day spans nothing.
        _plan_days_map = {k: v for k, v in (plan_data.get("plan") or {}).items()
                          if isinstance(v, dict) and (
                              str(_ci_get(v, "current_city", "city") or "").strip()
                              or any(_ci_get(v, *names) for names in
                                     (("flights", "flight"), ("hotel", "hotels"),
                                      ("attractions", "attraction"), ("cars", "car"))))}
        if getattr(meta, "days", 0) and meta.days > 0:
            checks.append(("days_covered", meta.days))
            if len(_plan_days_map) >= meta.days:
                matched += 1

        # ---- Lodging on EVERY night, not just once per city. ----
        # The per-city check below is satisfied by one hotel in a city, so a 3-night stay lodged for
        # a single night is currently task-perfect (task201 sleeps in Shanghai only on day2 though
        # the traveller lands 06:47 and is present day1). A traveller with no bed on a night they are
        # demonstrably in the city is not a valid itinerary — a core requirement, not a tuned
        # threshold, and a genuinely ADDITIONAL check (the fine-grained decomposition of the coarse
        # per-city one). Every day except the final checkout/return-home day needs a hotel; gold
        # books exactly this on all 533 feasible tasks (2,308 away-nights, 0 unlodged), so it is
        # gold-safe. Only fires when the task requires a hotel and the plan spans >= 2 days.
        if getattr(meta, "req_hotel", None):
            _day_items = sorted(
                ((k, v) for k, v in (plan_data.get("plan") or {}).items() if isinstance(v, dict)),
                key=lambda kv: _day_order(kv[0]))
            if len(_day_items) >= 2:
                for _dk, _dv in _day_items[:-1]:          # every night except the final checkout day
                    _h = _ci_get(_dv, "hotel", "hotels")
                    if isinstance(_h, list):
                        _h = _h[0] if _h else None
                    checks.append(("lodging_on_night", _dk))
                    if isinstance(_h, dict) and _ci_get(_h, "name", "hotel_name"):
                        matched += 1

        # One hotel per stay city. A three-city trip answered with a single hotel is not an
        # itinerary; only D0-src used to notice, so the skeleton still scored well overall.
        from cost_model import stay_cities_in_order as _stay
        _stay_cities = _stay(meta) or []
        if getattr(meta, "req_hotel", None) and _stay_cities:
            _booked_hotel_cities = {_fold(str(h.get("city") or "")) for h in plan_data["hotels"]}
            for _c in _stay_cities:
                checks.append(("lodging_in_city", _c))
                if _fold(_c) in _booked_hotel_cities:
                    matched += 1

        # ---- Substance: every stay city must actually be visited, not just slept in. ----
        # A 9-day, 3-city trip with ZERO attractions previously kept D0-key at 1.00 (sightseeing is
        # not a named constraint), so "book the flights, one hotel, and stop" was nearly free.
        # The requirement is per CITY, not per day: an itinerary may legitimately contain a rest or
        # checkout day, but travelling to a city and scheduling nothing there is not an itinerary.
        if getattr(meta, "req_attraction", None) is not None and _stay_cities:
            _att_cities = {_fold(str(a.get("city") or "")) for a in plan_data["attractions"]}
            for _c in _stay_cities:
                checks.append(("activity_in_city", _c))
                if _fold(_c) in _att_cities:
                    matched += 1
        
        # 检查酒店约束 (新增)
        if meta.req_hotel:
            # 酒店名称 (支持 name 或 hotel_name)
            req_name = meta.req_hotel.get("name") or meta.req_hotel.get("hotel_name")
            if req_name and req_name != "-":
                checks.append(("hotel_name", req_name))
                # 检查所有预订的酒店
                if any(h.get("name", "").lower() == req_name.lower() for h in plan_data["hotels"]):
                    matched += 1
        
        # 检查租车约束 (新增)
        if meta.req_car:
            # 车型
            req_type = meta.req_car.get("car_type") or meta.req_car.get("type")
            if req_type and req_type != "-":
                checks.append(("car_type", req_type))
                # 检查所有租车
                if any((c.get("type") or c.get("car_type") or "").lower() == req_type.lower() 
                       for c in plan_data["cars"]):
                    matched += 1
        
        # 检查景点约束 (新增)
        if meta.req_attraction:
            # 景点名称 (可能是字符串或列表)
            req_attrs = []
            raw_attr = meta.req_attraction.get("attraction_name") or meta.req_attraction.get("name")
            
            if isinstance(raw_attr, list):
                req_attrs = raw_attr
            elif isinstance(raw_attr, str) and raw_attr != "-":
                req_attrs = [raw_attr]
                
            for attr_name in req_attrs:
                if attr_name:
                    checks.append(("attraction_name", attr_name))
                    # 检查所有景点
                    if any(a.get("name", "").lower() == attr_name.lower() for a in plan_data["attractions"]):
                        matched += 1
        
        # 检查 Road Trip 约束 (新增): 必须每天都有车
        if meta.implicit_keywords and any(k.lower() in ["road trip", "roadtrip"] for k in meta.implicit_keywords):
            checks.append(("road_trip", "daily_car"))
            
            # 计算有车的实际天数
            days_with_car = 0
            plan_dict = plan_data.get("plan", {})
            
            # 遍历每一天检查是否有车
            # 注意: plan 的 key 可能是 "day 1", "Day 1" 等，这里直接遍历 values
            for day_content in plan_dict.values():
                if isinstance(day_content, dict):
                    # 检查是否有 car 字段且非空
                    car = day_content.get("car") or day_content.get("Car")
                    if car and car != "-":
                         days_with_car += 1
            
            # 只有当每一天都租车时才算满足 (允许 1 天的宽容度，比如最后一天去机场?)
            # 用户要求 "必须每天都有"，所以严格检查: days_with_car >= meta.days
            # 但考虑到 meta.days 可能包含离开的一天，有时最后一天不租车。
            # 为了稳健，我们要求覆盖率达到 plan 实际天数 或 meta.days
            target_days = meta.days
            if days_with_car >= target_days:
                matched += 1

        # D0-key is an ALL-OR-NOTHING hard constraint: an explicitly requested plan is satisfied
        # only if EVERY required element is present. Partial credit is misleading here---a plan
        # missing a mandated leg, entity, or day does not fulfil the request.
        if not checks:
            return None
        return 1.0 if matched == len(checks) else 0.0
    
    def score_d0_source(self, plan_data: dict, meta: QueryMeta) -> float:
        """D0-2: 数据一致性验证 (KB grounding)。

        A named entity is a FACT to check whether or not it carries a city: a hotel/attraction/car
        with a missing or unknown city cannot be verified, so it counts as checked-and-unmatched
        (a hallucination), not skipped. The old `if name and city` let a city-less fabrication
        escape the truthfulness score entirely, and cars were never verified at all (240 car tasks
        exempt). All four entity types are now grounded.
        """
        facts_checked = 0
        facts_matched = 0

        # 验证航班 —— existence AND schedule realism. A real flight number carries fixed times;
        # stating different departure/arrival times is a fabricated schedule that B3 would otherwise
        # accept at face value (it reads the plan's stated times for reachability). Both must hold
        # for the flight to count as truthful. Gold-safe: 1,334 gold flights, 0 time mismatches
        # against the v2 KB.
        for flight in plan_data["flights"]:
            fn = flight.get("flight_number")
            if fn:
                facts_checked += 1
                if (self.db.verify_flight(fn,
                        departure_city=flight.get("departure_city"),
                        arrival_city=flight.get("arrival_city"))
                        and self.db.flight_times_match(fn,
                            flight.get("departure_time"), flight.get("arrival_time"))):
                    facts_matched += 1

        # 验证酒店
        for hotel in plan_data["hotels"]:
            name = hotel.get("name")
            if name:
                facts_checked += 1
                city = hotel.get("city")
                if city and self.db.verify_hotel(name, city):
                    facts_matched += 1

        # 验证景点
        for attr in plan_data["attractions"]:
            name = attr.get("name")
            if name:
                facts_checked += 1
                city = attr.get("city")
                if city and self.db.verify_attraction(name, city):
                    facts_matched += 1

        # 验证租车 (新增)
        for car in plan_data["cars"]:
            ctype = car.get("car_type") or car.get("type")
            if ctype:
                facts_checked += 1
                city = car.get("city")
                if city and self.db.verify_car(ctype, city):
                    facts_matched += 1

        # ---- ZERO TOLERANCE. Truthfulness is binary per task. ----
        # An itinerary containing an entity that does not exist is not "95.8% correct" — it is
        # unusable: the traveller arrives to find no such hotel. As a graded ratio a fabricated
        # hotel cost only 4.2% of the category, and the more real entities a plan listed the
        # cheaper each lie became. So: any unverifiable entity zeroes the task, and a model's
        # Truthfulness is the FRACTION OF TASKS it completes without fabricating anything — binary
        # where it matters (per itinerary), continuous and discriminative where it is reported
        # (per model).
        #
        # Coverage is deliberately NOT folded in here any more; D0-keyword owns it (required flight
        # legs, days covered, lodging in every stay city, an activity in every stay city). That
        # keeps the two constructs separate: D0-src asks "did you make anything up", D0-key asks
        # "did you cover what was asked".
        self._last_d0_source_ratio = (facts_matched / facts_checked) if facts_checked else None
        if facts_checked > 0:
            return 1.0 if facts_matched == facts_checked else 0.0
        else:
            # [修改] 如果声称可行但没有任何可验证实体，给 0 分
            if plan_data["is_feasible"]:
                return 0.0
            else:
                return None  # 正确拒绝时跳过
    
    def score_d1_implicit(self, plan_data: dict, meta: QueryMeta) -> float:
        """D1: 隐式需求命中率 (v2 - 结构化设施匹配)
        
        基于 implicit_scoring.py 实现的结构化评分逻辑：
        - 设施匹配：资源的 amenities/facilities/extra_services 包含任意关键词即满分
        - 特殊情况：luxury travelers (酒店双维度)、foodie (餐厅评分) 等
        - 跳过检查：某些隐式需求对某些资源类型无要求
        """
        implicit_kw = meta.implicit_keywords
        if not implicit_kw:
            return None  # the TASK carries no persona: genuinely inapplicable

        # Which resource types the TASK requires — this, not what the plan happened to book,
        # decides which persona x resource cells are scored. A required type with no booking is an
        # unmet persona need (0), never a skipped cell.
        required_types = set()
        if meta.req_hotel:
            required_types.add("hotel")
        if meta.req_car:
            required_types.add("car")
        if meta.req_attraction:
            required_types.add("attraction")

        from cost_model import stay_cities_in_order
        return score_d1_implicit_v2(plan_data, implicit_kw, required_types,
                                    stay_cities_in_order(meta) or [])
    
    def score_d2_city(self, plan_data: dict, meta: QueryMeta) -> dict[str, float]:
        """D2: 城市维度评分"""
        expected_cities = []
        if meta.req_flight:
            arr = meta.req_flight.get("arrival_city", [])
            if isinstance(arr, list):
                expected_cities = arr
            elif arr:
                expected_cities = [arr]
        
        # A city counts as VISITED only if the plan actually arranges something there. Scoring this
        # off the free-text `current_city` label made D2 a string-copy metric: a submission with
        # zero bookings scored 1.000 by typing the city names into that field, and D2 had zero
        # variance across every content degradation (sd = 0.000), i.e. it measured nothing.
        booked_cities = []
        for group in ("hotels", "attractions", "cars"):
            for it in plan_data.get(group, []) or []:
                c = it.get("city") or it.get("city_name")
                if c:
                    booked_cities.append(str(c))
        for f in plan_data.get("flights", []) or []:
            c = f.get("arrival_city")
            if c:
                booked_cities.append(str(c))
        actual_cities = booked_cities
        # The itinerary order (for ordered tasks) still reads the declared day sequence, but only
        # over cities the plan actually books.
        declared_order = [c for c in plan_data.get("cities", []) if _fold(c) in {_fold(x) for x in booked_cities}]

        # 计算覆盖率和顺序匹配
        if not expected_cities:
            score = 1.0 if actual_cities else 0.0
        else:
            if meta.is_ordered:
                score = self._compute_city_order_score(expected_cities, declared_order)
            else:
                actual_folded = {_fold(c) for c in actual_cities}
                matched = sum(1 for c in expected_cities if _fold(c) in actual_folded)
                # ALL-OR-NOTHING: destination coverage is a hard constraint---visiting a subset of
                # the requested cities does not fulfil the request, so a missed city fails D2.
                score = 1.0 if matched == len(expected_cities) else 0.0
        
        # 确定维度
        # is_ordered = meta.is_ordered  # Ignored as per user request (confirmed no ordered queries)
        num_cities = meta.cities_count
        
        result = {}
        if num_cities == 1:
            result["d2_unord_single"] = score
        else:
            result["d2_unord_multi"] = score
        
        return result
    
    def _compute_city_order_score(self, expected: list[str], actual: list[str]) -> float:
        """计算城市顺序匹配分数（使用 LCS）
        
        Args:
            expected: 期望的城市顺序
            actual: 实际的城市顺序
        
        Returns:
            0-1 的分数，1 表示完全匹配顺序
        """
        if not expected:
            return 1.0
        if not actual:
            return 0.0
        
        # 使用动态规划计算最长公共子序列 (LCS)
        m, n = len(expected), len(actual)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if expected[i-1] == actual[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        # 分数 = LCS长度 / 期望城市数
        # 这样可以衡量有多少城市按正确顺序访问
        return lcs_length / len(expected)
    
    def compute_total_cost(self, plan_data: dict, meta: QueryMeta) -> float:
        """计算 plan 总成本 - 使用 Evidence (SandboxDB) 价格"""
        total = 0.0
        
        # ---- Air fare × party size. Every leg the TASK requires is billed, booked or not. ----
        # A required leg the plan omitted is charged the cheapest direct fare on that leg, so
        # dropping legs cannot buy budget compliance (the same omission-is-free hole the lodging
        # term had). Extra legs the plan invents are billed on top.
        required_legs = []
        if meta.req_flight:
            _dep = meta.req_flight.get("departure_city")
            _arr = meta.req_flight.get("arrival_city") or []
            if isinstance(_arr, str):
                _arr = [_arr]
            _chain = ([_dep] + [c for c in _arr if c]) if _dep else [c for c in _arr if c]
            if str(meta.req_flight.get("trip_type", "")).lower().startswith("round") and _dep and len(_chain) > 1:
                _chain = _chain + [_dep]
            required_legs = [(a, b) for a, b in zip(_chain, _chain[1:]) if a and b]

        _covered = set()
        for flight in plan_data["flights"]:
            fn = flight.get("flight_number")
            if not fn:
                continue
            db_price = self.db.get_flight_price(fn)
            if db_price is None:
                p = flight.get("price", 0)
                db_price = p if isinstance(p, (int, float)) else 0
            total += db_price * meta.person_num
            a, b = flight.get("departure_city"), flight.get("arrival_city")
            if a and b:
                _covered.add((_fold(str(a)), _fold(str(b))))

        for a, b in required_legs:
            if (_fold(a), _fold(b)) in _covered:
                continue
            floor_fare = self.db.cheapest_flight_price(a, b)
            if floor_fare is not None:
                total += floor_fare * meta.person_num
        
        # ---- Lodging: billed per STAY-CITY NIGHT, never per hotel row the plan happens to list ----
        # Every night of the itinerary is paid for. A stay city the plan booked no hotel in is billed
        # at that city's CHEAPEST rate — the same floor the generator's C_floor uses — so omitting a
        # booking can never come in under an "impossible" budget (this was refuting 18% of the
        # budget-infeasible labels). Billing per city also stops a hotel switch inside one city from
        # being double-charged, which the old per-hotel-row loop did.
        # A traveller pays the rate they actually sleep at, so each night is billed at ITS OWN rate,
        # read from the plan's day map. The previous `min()` per city billed every night in a city at
        # the cheapest rate the plan mentioned there: booking one token cheap night and the city's
        # most expensive hotel for all the others left the total completely unchanged and D3 at a
        # perfect 1.000 (measured). That `min` was an over-correction of a real bug — a per-hotel-row
        # loop that charged BOTH hotels for the whole stay when a plan switched hotels inside a city.
        # Per-night billing fixes the double-charge without granting the retroactive discount.
        _cities, _city_idx, _nights_by_city, _cardays_by_city, _base_n, _base_d = city_maps(meta)

        def _rate_of(hotel):
            if not isinstance(hotel, dict) or not _ci_get(hotel, "name", "hotel_name"):
                return None
            nm = _ci_get(hotel, "name", "hotel_name")
            cy = _ci_get(hotel, "city", "city_name")
            price = self.db.get_hotel_price(nm, cy)
            if price is None:
                p = _ci_get(hotel, "price_per_night", "price") or 0
                price = float(p) if isinstance(p, (int, float)) and p else None
            return None if price is None else (price, _city_idx.get(_fold(str(cy or ""))))

        _rates_by_city = {}   # city index -> [nightly rate, one per day the plan books there]
        for _dk in sorted((plan_data.get("plan") or {}).keys(), key=_day_order):
            _day = (plan_data.get("plan") or {}).get(_dk)
            if not isinstance(_day, dict):
                continue
            _h = _ci_get(_day, "hotel", "hotels")
            if isinstance(_h, list):
                _h = _h[0] if _h else None
            _r = _rate_of(_h)
            # A hotel outside every stay city neither reduces nor inflates the bill; a consistency
            # check, not the cost model, is the right place to punish that.
            if _r and _r[1] is not None:
                _rates_by_city.setdefault(_r[1], []).append(_r[0])

        for _i, _city in enumerate(_cities):
            _need = _nights_by_city[_i]
            _rs = _rates_by_city.get(_i, [])[:_need]
            _floor = self.db.cheapest_hotel_price(_city)   # imputed floor for an unbooked night
            _billable = sum(_rs) + max(0, _need - len(_rs)) * (_floor or 0)
            total += _billable * meta.rooms_count
        
        # 景点门票（每人）- 从 DB 查询价格
        for attr in plan_data["attractions"]:
            name = attr.get("name")
            city = attr.get("city")
            if name:
                db_price = self.db.get_attraction_price(name, city)
                if db_price is not None:
                    total += db_price * meta.person_num
                else:
                    # 尝试直接查找
                    df = self.db.get_attractions(city) if city else pd.DataFrame()
                    if not df.empty:
                        matches = df[df["attraction_name"].str.lower() == name.lower()]
                        if not matches.empty:
                            ticket = matches.iloc[0].get("ticket_price", 0)
                            if pd.notna(ticket):
                                total += float(ticket) * meta.person_num
        
        # ---- Car hire: billed per STAY-CITY CAR-DAY whenever the task requires a car. ----
        # Same rule as lodging: a required city with no car booked is charged that city's cheapest
        # qualifying rate, so skipping the rental cannot buy budget compliance, and two cars listed
        # for one city are billed once rather than twice.
        _car_rate_by_city = {}
        for car in plan_data["cars"]:
            car_type = car.get("type") or car.get("car_type")
            city = car.get("city")
            if not car_type:
                continue
            # Pass the plan's own price so the DB can tell which of the several cars this
            # (city, car_type) holds was booked; it still returns the KB's price, never the
            # plan's, so an understated price cannot buy a cheaper bill.
            rate = self.db.get_car_price(car_type, city, car.get("price_per_day"))
            if rate is None:
                p = car.get("price_per_day", 0)
                rate = float(p) if isinstance(p, (int, float)) and p else None
            if rate is None:
                continue
            _ci = _city_idx.get(_fold(city or ""))
            if _ci is None:
                continue
            prev = _car_rate_by_city.get(_ci)
            _car_rate_by_city[_ci] = rate if prev is None else min(prev, rate)

        # getattr, not attribute access: the query generator prices candidate itineraries with a
        # lightweight meta stand-in that carries only the flight/day fields, so the cost model must
        # tolerate a meta without the req_* slots.
        _req_car = getattr(meta, "req_car", None)
        if _req_car:
            _cap = None
            try:
                _cap = int(_req_car.get("capacity")) if _req_car.get("capacity") else None
            except (TypeError, ValueError):
                _cap = None
            for _i, _city in enumerate(_cities):
                rate = _car_rate_by_city.get(_i)
                if rate is None:
                    rate = self.db.cheapest_car_price(_city, _cap)
                if rate is not None:
                    total += rate * _cardays_by_city[_i]
        else:
            # No car required: only what the plan actually booked is billed.
            for _i, rate in _car_rate_by_city.items():
                total += rate * _cardays_by_city[_i]
        
        return total
    
    def score_d3_budget(self, plan_data: dict, meta: QueryMeta) -> tuple[float, float]:
        """D3: 预算权衡评分"""
        if meta.budget <= 0:
            return None, 0.0

        total_cost = self.compute_total_cost(plan_data, meta)
        # D3 asks "did you deliver an itinerary that respects the budget", so a submission that
        # books NOTHING scores 0 rather than a perfect exp(0). The imputed market floor added to
        # compute_total_cost exists to price omissions *inside a real plan* — it must not turn an
        # empty submission into a budget-compliant one. (Before this gate an empty plan scored
        # D3 = 1.0 because the floor happened to fit the budget.)
        if not _plan_books_anything(plan_data):
            return 0.0, total_cost
        if total_cost <= 0:
            return 0.0, 0.0

        delta_b = max(0, (total_cost - meta.budget) / meta.budget)
        score = math.exp(-self.beta * delta_b)

        return score, total_cost
    
    def score_d4_impossible(self, plan_data: dict, meta: QueryMeta) -> Optional[float]:
        """D4: 不可行任务处理评分 (双向评估)
        
        对于 impossible=True 的查询：
          - LLM 正确拒绝 (is_feasible=False) 且无幻觉 → 1.0
          - LLM 正确拒绝但有幻觉 → 0.0
          - LLM 错误生成计划 → 0.0
          
        对于 impossible=False 的查询：
          - LLM 正确生成计划 (is_feasible=True) → 1.0
          - LLM 错误拒绝 (is_feasible=False) → 0.0
        """
        is_feasible = plan_data["is_feasible"]
        
        if meta.impossible:
            # 不可行任务：应该被拒绝
            if not is_feasible:
                # 正确拒绝，检查是否有幻觉
                has_hallucination = False
                
                for flight in plan_data["flights"]:
                    fn = flight.get("flight_number")
                    if fn and not self.db.verify_flight(fn):
                        has_hallucination = True
                        break
                
                if not has_hallucination:
                    for hotel in plan_data["hotels"]:
                        name = hotel.get("name")
                        city = hotel.get("city")
                        # A named entity the plan refuses to locate is UNVERIFIABLE, which is a
                        # hallucination for scoring purposes — not a free pass. The old
                        # `name and city` guard let a refusal padded with "Totally Fake Hotel 999"
                        # keep D4 = 1.0 simply by omitting the city field.
                        if name and (not city or not self.db.verify_hotel(name, city)):
                            has_hallucination = True
                            break
                
                if not has_hallucination:
                    for attr in plan_data["attractions"]:
                        name = attr.get("name")
                        city = attr.get("city")
                        if name and (not city or not self.db.verify_attraction(name, city)):
                            has_hallucination = True
                            break

                if not has_hallucination:
                    # Cars were the one entity type this scan skipped, so a refusal padded with an
                    # invented car ("Hovercraft in Paris") kept D4 = 1.0 and took the whole
                    # Infeasibility category with it — while the same fabrication in a hotel,
                    # attraction or flight scored 0. D0-src already treats a car as an entity that
                    # can be invented; this makes the refusal path agree with it.
                    for car in plan_data["cars"]:
                        ctype = car.get("car_type") or car.get("type")
                        city = car.get("city")
                        if ctype and (not city or not self.db.verify_car(ctype, city)):
                            has_hallucination = True
                            break

                if has_hallucination:
                    return 0.0
                # A refusal must DIAGNOSE, not just decline. ALL-OR-NOTHING: correctly handling an
                # infeasible task means naming the RIGHT typed cause; a bare "I refuse" or a wrong
                # cause has not solved the typed-infeasibility task, so it scores 0, not partial.
                truth = true_impossibility_class(meta, self.db)
                stated = stated_impossibility_class(plan_data.get("refusal_reason"))
                return 1.0 if stated == truth else 0.0
            else:
                # 错误地提供了计划（应该拒绝）
                return 0.0
        else:
            # 可行任务：应该生成计划
            # D4 scores the FEASIBILITY DECISION. On a feasible task the correct decision is not
            # merely "did not say no" — it is "delivered an itinerary". A submission that claims
            # is_feasible=True and books nothing is a de-facto refusal dressed as an acceptance, and
            # used to collect D4 = 1.0 (and with B2/B3 returning None, a perfect 25% Reasoning
            # category) for an empty plan.
            if is_feasible and _plan_books_anything(plan_data):
                return 1.0
            return 0.0
    
    def score_d5_retry_robustness(self, tool_call_count: int, meta: QueryMeta,
                                  plan_data: dict = None, total_tokens: int = None,
                                  is_valid: bool = None) -> float:
        """D5: 重试稳健性评分（工具调用效率）
        
        基于实际工具调用次数与最小理论次数的差距计算效率分数。
        
        评分公式：
        - 最小调用次数 = 理论最少需要的工具调用次数
        - 超出次数 = max(0, tool_call_count - min_calls)
        - 分数 = max(0, 100 - 扣分梯度 × 超出次数) / 100
        
        扣分梯度根据 max_tool_calls 动态调整：
        - max_tool_calls = 15: 每多一次扣 10 分（15次用完扣150分，归零）
        - max_tool_calls = 10: 每多一次扣 15 分（10次用完扣150分，归零）
        
        Args:
            tool_call_count: 实际工具调用次数
            meta: 查询元数据
        
        Returns:
            0-1 的分数，1 表示最优效率；None 表示没有可信的调用计数（不评分）
        """
        # FAIL CLOSED: a missing/zero tool_call_count is "not measured", NOT "perfectly efficient".
        # The old default of 0 gave 0 < min_calls -> extra_calls 0 -> 1.0, so a plan submitted with
        # no call count silently maxed the entire 25% Efficiency category.
        if not tool_call_count or tool_call_count <= 0:
            return None

        # Efficiency is credited only for a DELIVERED itinerary. Spending one tool call and
        # submitting nothing is not efficient, it is not doing the task — yet it used to score
        # D5 = 1.0 and hand a do-nothing agent the whole 25% Efficiency category.
        if plan_data is not None and not _plan_books_anything(plan_data):
            return 0.0

        # EFFICIENCY IS CONDITIONAL ON A VALID ANSWER. Producing an unusable itinerary cheaply is
        # not efficiency — it is just cheap. A run that fails a hard validity gate (invented an
        # entity, ignored a mandated booking) earns no efficiency credit at all.
        if is_valid is False:
            return 0.0

        # Minimum tool calls under the agent protocol. The pipeline counts submit_plan as a call
        # (react_engine tool_call_count++ fires before the submit check), and the system prompt
        # mandates querying hotels/attractions/cars ONE CITY AT A TIME ("search_hotels(city=...)
        # per city"), so the true minimum is per-city, not a single batched call. The old model
        # charged 1 call per domain and cities_count flight legs, which under-counted every
        # multi-city plan — a protocol-following agent was then docked for calls it had to make.
        k = meta.cities_count if getattr(meta, 'cities_count', None) else 1

        # 航班: 航段数 = 单程 k 段, 往返 k+1 段 (回程)
        if meta.req_flight:
            round_trip = str(meta.req_flight.get("trip_type", "one_way")).lower() in ("round", "round_trip", "round trip")
            min_flight_calls = (k + 1) if round_trip else k
        else:
            min_flight_calls = 0

        # 酒店/景点/租车: 每个到访城市各查一次
        min_hotel_calls = k if meta.req_hotel else 0
        min_attraction_calls = k if meta.req_attraction else 0
        min_car_calls = k if meta.req_car else 0

        # 提交计划: 1次 (被 pipeline 计入 tool_call_count)
        min_submit = 1

        min_calls = min_flight_calls + min_hotel_calls + min_attraction_calls + min_car_calls + min_submit
        min_calls = max(min_calls, 2)
        
        # ---- TWO-SIDED. Efficiency means reaching the goal with few calls, not making few calls. ----
        # The one-sided form scored 1.0 for ANY count at or below min_calls, so an agent that did
        # half the required searching was rewarded: a 3-city/9-day task needing ~10 calls was
        # answered with 5 (one hotel for three cities, zero attractions) and still took a perfect
        # 25% Efficiency category. Under-calling means the itinerary cannot have been grounded, so
        # it is penalised on the same gradient as over-calling.
        max_tool_calls = self.config.get("max_tool_calls", 15)
        penalty_per_call = 150 / max_tool_calls          # ~10 points per surplus call

        over = max(0, tool_call_count - min_calls)
        under = max(0, min_calls - tool_call_count)

        # Under-work is charged relative to the work the task demanded, so skipping half the
        # required searches costs half the category rather than a fixed per-call amount.
        under_ratio = (under / min_calls) if min_calls else 0.0
        efficiency = max(0.0, 100 - penalty_per_call * over) / 100.0
        efficiency *= max(0.0, 1.0 - under_ratio)

        # ---- Token cost. Calls alone are a weak proxy: the real bill is context. ----
        # Our lossless notebook resends the full transcript each turn, so tokens grow with the
        # number of calls; measured on real runs this is ~6k tokens per tool call plus a fixed
        # prompt overhead. The budget is TASK-DERIVED (never cohort-relative, which would make the
        # metric depend on which models happen to be in the run and break reproducibility), and
        # generous — it is meant to catch a model that loops for 40 turns, not normal operation.
        if total_tokens:
            _T = max(1, min_calls)
            budget = TOKEN_BUDGET_BASE * _T + TOKEN_BUDGET_PER_CALL * _T * (_T + 1) / 2
            overrun = max(0.0, total_tokens / budget - 1.0)
            efficiency *= math.exp(-TOKEN_OVERRUN_BETA * overrun)
        return efficiency
    

    
    def score_b2_opening_hours(self, plan_data: dict, meta: QueryMeta = None) -> float:
        """B2: 营业时间合规评分。

        TASK-DERIVED DENOMINATOR. The compliant count is divided by
        `max(len(visits), number of stay cities)`, not by `len(visits)` alone. With a purely
        submission-chosen denominator, deleting every visit but one correctly-timed one took B2 to
        1.000 — and with B3 it took the whole Reasoning category to 1.000 — on a 9-day 3-city trip.
        That is a construct failure whether or not it pays: Reasoning is the capability this
        benchmark claims to measure, so it must not be maximisable by scheduling almost nothing.

        The floor is exactly the stay-city count because that is the largest value the reference
        itineraries survive: measured over all 533 feasible gold plans, the minimum visits-per-stay-
        city ratio is exactly 1.00 at every trip size (1-city min 1 visit, 2-city min 3, 3-city
        min 6). A higher floor — per day, say — would fail gold, which schedules as little as one
        visit for a single-city trip. Where the task genuinely asks for no more, one correctly timed
        visit is not gaming; it matches the reference.

        A visit is COMPLIANT only if it is verifiable and correct: the attraction exists in the KB,
        the plan states a parseable `visit_start`, and that time falls inside the opening hours.
        Every other case is a violation.

        This closes two loopholes. `is_time_in_range` is fail-open — it returns True when there is
        no opening-hours data, and again when the visit time cannot be parsed — and the old loop
        only counted a violation `if name and city and visit_start`, while the denominator stayed
        `len(visits)`. So a plan that simply omitted `visit_start`, or that invented attractions the
        KB has never heard of, scored a perfect 1.0. The dimension rewarded withholding information.
        """
        visits = plan_data["attractions"]
        if not visits:
            # APPLICABILITY IS A PROPERTY OF THE TASK, NOT OF THE SUBMISSION. B2 is only reached on
            # a feasible task, i.e. one that asked for an itinerary, so scheduling no visits at all
            # is a FAILURE of this dimension, not an exemption from it. Returning None here removed
            # the metric from the average, which made "book nothing" outscore "book and mis-schedule"
            # (deleting every attraction cost 0.005 overall and left Reasoning at a perfect 1.000).
            return 0.0

        violations = 0
        for attr in visits:
            name = attr.get("name")
            city = attr.get("city")
            visit_start = attr.get("visit_start")

            if not name or not city:
                violations += 1                      # unattributable visit
                continue
            start_m = parse_time(visit_start)
            if start_m is None:
                violations += 1                      # no schedule stated: absence, not compliance
                continue
            open_hours = self.db.get_open_hours(name, city)
            if not open_hours or open_hours == "-":
                violations += 1                      # entity absent from the KB: unverifiable
                continue

            # The visit must fit ENTIRELY inside the opening hours and be long enough to actually
            # happen. Checking only `visit_start` (the old behaviour) passed visits that ran hours
            # past closing, and passed zero-length visits — while the agent prompt asks for both a
            # start and an end covering `duration_of_visit`, so not checking them was a free pass.
            end_m = parse_time(attr.get("visit_end"))
            if end_m is None or end_m <= start_m:
                violations += 1                      # no/degenerate visit window
                continue

            open_m, close_m = _parse_hours_range(open_hours)
            if open_m is None or close_m is None:
                violations += 1                      # unparseable hours: unverifiable, not compliant
                continue
            if start_m < open_m or end_m > close_m:
                violations += 1                      # starts before opening or runs past closing
                continue

            # Now that `get_visit_duration` parses ("2 hours" -> 2.0) this check finally runs at all.
            # It enforces only min(KB duration, 60 min), NOT the full KB duration: 898 of gold's
            # 3,431 timed visits (26.2%) are shorter than the KB figure, so the strict form would
            # fail the reference itinerary on a quarter of its visits — that is a generator debt to
            # settle before the full check can ship. At the 60-minute floor gold is clean (its
            # shortest visit is 73 min, 0/3,431 violations) while a token 5-minute stop at a
            # three-hour museum, previously fully compliant, now counts as the violation it is.
            need = self.db.get_visit_duration(name, city)
            if need is not None and (end_m - start_m) + 1e-9 < min(need * 60, 60):
                violations += 1                      # not enough time booked to actually visit

        denom = len(visits)
        if meta is not None:
            from cost_model import stay_cities_in_order
            denom = max(denom, len(stay_cities_in_order(meta) or []))
        return (len(visits) - violations) / denom
    
    def score_b3_spatiotemporal(self, plan_data: dict) -> float:
        """B3: 时空可行性评分（按天独立计算，不跨天检查）"""
        plan = plan_data.get("plan", {})
        
        total_violations = 0
        total_transitions = 0
        
        for day_key in sorted(plan.keys()):
            day_data = plan.get(day_key, {})
            if not isinstance(day_data, dict):
                continue
            
            # 构建当天事件序列
            day_events = []
            
            # A booking that must be scheduled but carries no parseable time is a violation, not a
            # skip. Previously such entries simply produced no event, so a plan with no times at all
            # left `total_transitions == 0` and the whole dimension returned None — omitting the
            # schedule made B3 disappear instead of failing it.
            untimed = 0

            # 航班
            flights = _ci_get(day_data, "flights", "flight") or []
            if isinstance(flights, list):
                for f in flights:
                    if not isinstance(f, dict):
                        continue
                    if parse_time(f.get("departure_time")) is None:
                        untimed += 1
                        continue
                    _fn = f.get("flight_number")
                    day_events.append({
                        "type": "flight",
                        "start": f.get("departure_time"),
                        "end": f.get("arrival_time"),
                        "city": f.get("arrival_city"),
                        "name": _fn,
                        # a flight BEGINS at its departure airport and ENDS at its arrival airport
                        "loc_start": self.db.get_flight_endpoint(_fn, "departure"),
                        "loc_end": self.db.get_flight_endpoint(_fn, "arrival"),
                    })

            # 景点
            attractions = _ci_get(day_data, "attractions", "attraction") or []
            if isinstance(attractions, list):
                for a in attractions:
                    if not isinstance(a, dict):
                        continue
                    if parse_time(a.get("visit_start")) is None:
                        untimed += 1
                        continue
                    day_events.append({
                        "type": "attraction",
                        "start": a.get("visit_start"),
                        "end": a.get("visit_end"),
                        "city": a.get("city"),
                        "name": a.get("name")
                    })
            
            # 酒店（入住）
            hotel = _ci_get(day_data, "hotel", "hotels")
            if isinstance(hotel, list):
                hotel = hotel[0] if hotel else None
            _ci = _ci_get(hotel, "check_in", "checkin") if isinstance(hotel, dict) else None
            if _ci:
                if parse_time(_ci) is None:
                    # SAME RULE AS FLIGHTS AND ATTRACTIONS. An unparseable check_in used to be
                    # admitted as an event anyway; `parse_time(...) or 0` then sorted it to midnight,
                    # so it became the FIRST event of the day, and the pair it opened was dropped by
                    # the `end_time is None` guard — removed from the numerator AND the denominator.
                    # Measured on 308 real runs: 42.3% of hotel entries carry an unparseable value
                    # (models write the DAY — "day1" x120, "day4" x37 — because nothing documented
                    # the format), which swallowed 30.3% of all the pairs B3 walks and floored 25.5%
                    # of itineraries to B3=0.0 having run zero checks. Gold has 0 unparseable
                    # check-ins, so failing closed here cannot touch the reference.
                    untimed += 1
                else:
                    day_events.append({
                        "type": "hotel",
                        "start": _ci,
                        "end": None,
                        "city": _ci_get(hotel, "city", "city_name"),
                        "name": _ci_get(hotel, "name", "hotel_name")
                    })

            # every un-scheduled booking counts against the day
            total_violations += untimed
            total_transitions += untimed

            if len(day_events) < 2:
                continue  # 当天可排事件太少，没有衔接可判

            # Every event now has a parseable start, so the sort key is total — nothing can land at
            # key 0 by accident and displace the day's real first event.
            day_events.sort(key=lambda e: parse_time(e.get("start")))

            # 检查当天相邻事件的可行性
            for i in range(len(day_events) - 1):
                e1, e2 = day_events[i], day_events[i+1]

                # 获取时间间隔
                start_time = parse_time(e2.get("start"))

                # FAIL CLOSED on a withheld end time. A flight and a visit both occupy an interval,
                # so falling back to `start` modelled them as instantaneous and handed the plan free
                # travel slack for a field it simply declined to state. Measured: on deliberately
                # over-packed itineraries, dropping only `arrival_time` (which B2 never inspects, so
                # it was entirely free) lifted B3 from 0.447 to 0.483 and the overall score by
                # +0.005 — withholding information paid. A hotel check-in is a genuine point event
                # and keeps the fallback.
                if e1.get("type") in ("flight", "attraction"):
                    end_time = parse_time(e1.get("end"))
                    if end_time is None:
                        total_transitions += 1
                        total_violations += 1
                        continue
                else:
                    end_time = parse_time(e1.get("end") or e1.get("start"))

                if end_time is None or start_time is None:
                    continue

                total_transitions += 1
                delta_t = start_time - end_time  # 分钟

                # Distance is measured from where e1 LEAVES YOU to where e2 PICKS YOU UP.
                # For everything but a flight those are the same point; for a flight they are the
                # arrival and departure airports respectively.
                coord1 = e1.get("loc_end")
                if coord1 is None:
                    coord1 = self.db.get_coordinates(e1["name"], e1["type"], e1["city"])
                coord2 = e2.get("loc_start")
                if coord2 is None:
                    coord2 = self.db.get_coordinates(e2["name"], e2["type"], e2["city"])

                if not coord1 or not coord2:
                    # an endpoint the KB cannot locate is unverifiable, not a free pass
                    total_violations += 1
                    continue

                distance = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
                if delta_t < get_min_travel_time(distance):
                    total_violations += 1
        
        if total_transitions == 0:
            # Same rule as B2: reached only on a feasible task, so producing no schedulable pair of
            # events is a failed itinerary, not an inapplicable dimension. Returning None let a plan
            # delete B3 benchmark-wide simply by putting each event under its own day key.
            return 0.0
        
        return 1 - total_violations / total_transitions
    
    def score_query(self, plan_output: dict, meta: QueryMeta, tool_call_count: int = None,
                    total_tokens: int = None, elapsed_sec: float = None) -> ScoreResult:
        """对单个查询进行全面评分"""
        plan_data = self.extract_plan_data(plan_output)
        
        result = ScoreResult(
            query_index=0,
            is_feasible=plan_data["is_feasible"],
            attraction_count=len(plan_data["attractions"])
        )
        # Run metadata: recorded unconditionally, BEFORE any early-return branch, so a correct
        # refusal still reports what it cost to produce.
        result.total_tokens = total_tokens
        result.task_impossible = bool(getattr(meta, "impossible", False))
        result.elapsed_sec = elapsed_sec
        
        # [新增] 检测 False Negative: 可行任务被错误拒绝
        # 如果任务是可行的 (impossible=False) 但 LLM 拒绝了 (is_feasible=False)
        # 则所有维度给 0 分，严厉惩罚这种错误
        if not meta.impossible and not plan_data["is_feasible"]:
            result.d0_keyword = 0.0
            result.d0_source = 0.0
            # D1 applicability is task-derived: a task with NO persona is inapplicable (None) even
            # when wrongly refused---setting it 0.0 here made the D1 denominator submission-dependent
            # (a model that refused more no-persona feasible tasks was averaged over more tasks).
            result.d1_implicit = 0.0 if meta.implicit_keywords else None
            # D2: 根据城市数量设置对应维度为 0
            if meta.cities_count == 1:
                result.d2_unord_single = 0.0
            else:
                result.d2_unord_multi = 0.0
            result.d3_budget = 0.0 if meta.budget > 0 else None
            result.d4_impossible = 0.0  # 核心惩罚
            result.d5_retry = 0.0
            result.b2_opening_hours = 0.0
            result.b3_spatiotemporal = 0.0
            result.total_cost = 0.0
            result.task_success = False
            return result
        
        # [O5 / task-derived applicability] False Positive: an INFEASIBLE task the agent wrongly
        # answered with a plan (impossible=True, is_feasible=True). The failure is fully captured by
        # D4=0.0 (it should have refused) and by task_success/task_perfect=False. The feasible-only
        # dimensions are OUT OF SCOPE on an infeasible task, so they stay None (their ScoreResult
        # default) — NOT 0.0. Scoring them 0.0 here (a) double-penalised the non-refusal (once on D4,
        # again on Sat/Tru/Exe) and (b) leaked the infeasible task into the feasible-dimension
        # denominators, making n_scored submission-dependent — violating the invariant that
        # applicability is a property of the TASK, identical for every model (see
        # compute_aggregate_scores and §4). Symmetric to the correct-refusal branch below, which
        # already scores only D4.
        if meta.impossible and plan_data["is_feasible"]:
            result.d4_impossible = 0.0  # failed to refuse an impossible request
            result.total_cost = 0.0
            result.task_success = False
            return result

        # [O5] 正确拒绝: 不可行任务且模型拒绝 (impossible=True, is_feasible=False)。
        # 只有 D4 (拒绝的正确性/无幻觉) 适用; 其余维度对空计划无意义, 一律 None —— 与
        # D0-src/D1/B2/B3 早已实现的 "正确拒绝时跳过" 一致。此前 D0-key/D2/D3 走普通路径,
        # 在空计划上算出 0/0/1.0, 把完美 gold 的聚合拉到 0.666/0.697/0.641, 并让 "拒绝+塞一个
        # 真实体" 同时拿 D4=1 和 D0-src=1 反超 gold。
        if meta.impossible and not plan_data["is_feasible"]:
            result.d4_impossible = self.score_d4_impossible(plan_data, meta)
            # TASK-DERIVED SUCCESS. Set on this branch too: `fully_valid` is only reachable further
            # down, so a correct refusal used to leave it None and drop out of `fully_valid_rate`'s
            # denominator entirely. That made the denominator a function of the model — a model that
            # correctly refused all 267 infeasible tasks was rated on 533, one that never refused on
            # 800 — which is exactly the model-dependent applicability the scorer forbids everywhere
            # else. Success on an infeasible task IS the correct refusal, so score it as one.
            result.task_success = (result.d4_impossible == 1.0)
            return result

        # ---- Hard validity gates (binary per itinerary) ----
        _valid, _reasons = plan_validity_gates(plan_data, meta, self.db)
        result.fully_valid = _valid
        result.invalid_reasons = _reasons or None
        # An infeasible task answered with an itinerary is a failed task whatever the itinerary
        # looks like: the required output was a refusal.
        result.task_success = (False if meta.impossible
                               else bool(_valid) and plan_data["is_feasible"])
        # D0
        result.d0_keyword = self.score_d0_keyword(plan_data, meta)
        result.d0_source = self.score_d0_source(plan_data, meta)
        result.d0_source_ratio = getattr(self, "_last_d0_source_ratio", None)
        
        # D1
        result.d1_implicit = self.score_d1_implicit(plan_data, meta)
        
        # D2
        d2_scores = self.score_d2_city(plan_data, meta)
        result.d2_unord_single = d2_scores.get("d2_unord_single")
        result.d2_unord_multi = d2_scores.get("d2_unord_multi")
        
        # D3
        d3_score, total_cost = self.score_d3_budget(plan_data, meta)
        result.d3_budget = d3_score
        result.total_cost = total_cost
        
        # D4
        result.d4_impossible = self.score_d4_impossible(plan_data, meta)
        
        # D5: 工具调用效率
        result.d5_retry = self.score_d5_retry_robustness(
            tool_call_count, meta, plan_data, total_tokens=total_tokens, is_valid=_valid)
        
        # CCR
        result.b2_opening_hours = self.score_b2_opening_hours(plan_data, meta)
        result.b3_spatiotemporal = self.score_b3_spatiotemporal(plan_data)
        
        return result


# ============ 汇总统计 ============
def compute_aggregate_scores(results: list[ScoreResult]) -> dict:
    """计算汇总统计"""
    def avg_non_none(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None

    def category_conjunction(attrs):
        """ALL-OR-NOTHING at the category level: a task PASSES a category iff EVERY applicable
        (non-None) dimension in it scores 1.0; the category score is the fraction of tasks (with at
        least one applicable dimension) that pass. This makes a category an honest "was this whole
        requirement group met?" rather than a mean that one strong dimension can prop up."""
        num = den = 0
        for r in results:
            vals = [getattr(r, a) for a in attrs if getattr(r, a) is not None]
            if not vals:
                continue
            den += 1
            if all(v >= 1.0 for v in vals):
                num += 1
        return (num / den) if den else None
    
    agg = {
        "d0_keyword": avg_non_none([r.d0_keyword for r in results]),
        "d0_source": avg_non_none([r.d0_source for r in results]),
        "d1_implicit": avg_non_none([r.d1_implicit for r in results]),
        "d2_unord_single": avg_non_none([r.d2_unord_single for r in results]),
        "d2_unord_multi": avg_non_none([r.d2_unord_multi for r in results]),
        "d3_budget": avg_non_none([r.d3_budget for r in results]),
        "d4_impossible": avg_non_none([r.d4_impossible for r in results]),
        "d5_retry": avg_non_none([r.d5_retry for r in results]),
        "b2_opening_hours": avg_non_none([r.b2_opening_hours for r in results]),
        "b3_spatiotemporal": avg_non_none([r.b3_spatiotemporal for r in results]),
    }
    
    # ========== 四大类评分 (25% 权重) ==========
    # 1. 约束满足 (Satisfaction): D0-keyword, D1-implicit, D2-city, D3-budget
    # Constraint Satisfaction is ALL-OR-NOTHING: a task is satisfied only if EVERY explicit element,
    # EVERY implicit persona need, EVERY required city, and the budget are all met. Partial credit
    # is misleading here---a plan that meets three of four requirement groups still fails to give the
    # traveller what they asked for. (The agent is explicitly told the persona needs are scored.)
    cat_satisfaction = category_conjunction(
        ("d0_keyword", "d1_implicit", "d2_unord_single", "d2_unord_multi", "d3_budget"))
    
    # 2. 数据真实性 (Truthfulness): D0-source
    cat_truthfulness = agg["d0_source"]
    
    # 3. 规划合理性 (Reasoning): B2, B3 — scheduling quality only.
    # D4 is REPORTED SEPARATELY rather than averaged in here. On a feasible task it is ~constant
    # (delivering anything substantive scores 1.0), so folding it in propped the category up and
    # diluted real scheduling failures: an itinerary that put every visit outside opening hours
    # still cleared 0.5 on Reasoning purely because D4 handed it a free 1.0.
    # Executability is ALL-OR-NOTHING: a plan is executable only if EVERY attraction visit is within
    # opening hours AND EVERY same-day transition is physically reachable in the time allotted. One
    # infeasible hop makes a day the traveller cannot actually follow.
    cat_reasoning = category_conjunction(("b2_opening_hours", "b3_spatiotemporal"))
    # Infeasibility detection is its own headline number over the 267 infeasible tasks — a third of
    # the benchmark that previously drove only 2.78% of the score.
    #
    # SCOPED TO THE INFEASIBLE TASKS, matching what this comment has always claimed. The
    # implementation used to average d4 over ALL tasks, and on a feasible task d4 is 1.0 for merely
    # delivering a plan — so 533 near-automatic 1.0s were folded into a category named
    # "infeasibility detection". Measured on the full run, that inflated the weak tail most:
    # nova-2-lite read 0.665 when its true detection rate over the 267 was 0.303, qwen3 0.530 vs
    # 0.262 — while the strong refusers barely moved (kimi 0.943 vs 0.959). Gold scores d4 = 1.0000
    # on all 267, so the scoping cannot touch the reference. d4 on feasible tasks still feeds
    # task_success and the per-dimension mean; it just no longer pads this category.
    _imp_d4 = [r.d4_impossible for r in results
               if getattr(r, "task_impossible", None) and r.d4_impossible is not None]
    if _imp_d4:
        agg["cat_infeasibility"] = sum(_imp_d4) / len(_imp_d4)
    else:
        # No flagged results (e.g. legacy inputs without the flag, or a feasible-only slice):
        # fall back to the unscoped mean rather than dropping the category.
        agg["cat_infeasibility"] = agg["d4_impossible"]
    
    # 4. Agent 效率 (Efficiency): D5-retry
    cat_efficiency = agg["d5_retry"]
    
    # 添加到结果
    agg["cat_satisfaction"] = cat_satisfaction
    agg["cat_truthfulness"] = cat_truthfulness
    agg["cat_reasoning"] = cat_reasoning
    agg["cat_efficiency"] = cat_efficiency
    
    # ---- Per-dimension denominators, reported. ----
    # Applicability is task-derived (see score_query), so these counts are a property of the
    # BENCHMARK and must be identical for every model. Reporting them lets a reader verify that two
    # models were averaged over the same tasks — cross-model means were previously computed over
    # model-dependent subsets and were therefore not comparable.
    agg["n_scored"] = {k: sum(1 for r in results if getattr(r, k) is not None)
                       for k in ("d0_keyword", "d0_source", "d1_implicit", "d2_unord_single",
                                 "d2_unord_multi", "d3_budget", "d4_impossible", "d5_retry",
                                 "b2_opening_hours", "b3_spatiotemporal")}
    agg["n_results"] = len(results)

    # Headline binary number, HumanEval-style: per itinerary the gates are pass/fail, but the rate
    # across the benchmark is continuous and discriminative.
    _ts = [r for r in results if getattr(r, "task_success", None) is not None]
    agg["task_success_rate"] = (sum(1 for r in _ts if r.task_success) / len(results)) if results else None

    # HEADLINE. `task_success_rate` above is a FLOOR: it asks only whether the submission is
    # structurally valid and got the feasible/infeasible call right, and the frontier models clear
    # it on nearly every task. This asks the question the benchmark is actually about — is the
    # itinerary correct on every dimension the TASK brings into scope — and it is the direct
    # analogue of HumanEval pass@1 / SWE-bench resolve rate: binary per task, continuous across the
    # benchmark. The difficulty comes from conjunctive aggregation at the task level, not from any
    # tuned threshold; empirically the bar's exact position is not arbitrary either, since requiring
    # >= 0.95 instead of == 1.0 leaves every model's rate unchanged.
    _PERFECT_DIMS = ("d0_keyword", "d0_source", "d1_implicit", "d2_unord_single", "d2_unord_multi",
                     "d3_budget", "d4_impossible", "b2_opening_hours", "b3_spatiotemporal")
    def _perfect(r):
        vals = [v for v in (getattr(r, d, None) for d in _PERFECT_DIMS) if v is not None]
        if not vals or min(vals) < 1.0:
            return False
        return getattr(r, "fully_valid", None) is not False
    agg["task_perfect_rate"] = (sum(1 for r in results if _perfect(r)) / len(results)) if results else None

    # DISAGGREGATE the headline by task feasibility. The all-800 rate blends two very different
    # regimes: constructing an executable itinerary (hard) and correctly refusing an impossible one
    # (near-saturated — the frontier model scores 97.0% here vs 48.6% on feasible planning). Blending
    # them flatters the headline and lets a model that mostly excels at refusal (deepseek) outrank a
    # better planner. Reporting the two separately is parameter-free, needs no scorer or data change,
    # and is gold-safe by construction. task_perfect_feasible is the honest planning headline.
    _feas = [r for r in results if getattr(r, "task_impossible", None) is not True]
    _infe = [r for r in results if getattr(r, "task_impossible", None) is True]
    agg["task_perfect_feasible"] = (sum(1 for r in _feas if _perfect(r)) / len(_feas)) if _feas else None
    agg["task_perfect_infeasible"] = (sum(1 for r in _infe if _perfect(r)) / len(_infe)) if _infe else None
    agg["n_feasible"] = len(_feas)
    agg["n_infeasible"] = len(_infe)

    # Companion headline: on what fraction of tasks did the model satisfy EVERY explicitly stated
    # requirement (d0_keyword == 1.0)? Explicit requirements are spelled out in the query text, so
    # partial credit is reported in the graded diagnostic but the headline companion treats them as
    # all-or-nothing — you either honoured what the user asked for, or you did not.
    _ex = [r for r in results if r.d0_keyword is not None]
    agg["explicit_full_rate"] = (sum(1 for r in _ex if r.d0_keyword >= 0.9999) / len(_ex)) if _ex else None
    # Retained for continuity with earlier runs; note its denominator is model-dependent and it is
    # NOT the headline number. `task_success_rate` above is.
    _gated = [r for r in results if getattr(r, "fully_valid", None) is not None]
    agg["fully_valid_rate"] = (sum(1 for r in _gated if r.fully_valid) / len(_gated)) if _gated else None
    _tok = [r.total_tokens for r in results if getattr(r, "total_tokens", None)]
    _sec = [r.elapsed_sec for r in results if getattr(r, "elapsed_sec", None)]
    agg["avg_total_tokens"] = (sum(_tok) / len(_tok)) if _tok else None
    agg["avg_elapsed_sec"] = (sum(_sec) / len(_sec)) if _sec else None

    # ---- Weighted overall: NO renormalisation. ----
    # Renormalising over "available" categories meant an UNMEASURED category raised the score: two
    # otherwise identical runs scored 1.0000 vs 0.7500 purely on whether tool counts were logged.
    # A missing category is a defect in the run, so the headline is withheld and named instead.
    # FIVE equally-weighted categories (20% each). Infeasibility detection is now one of them:
    # the 267 infeasible tasks are 33.4% of the benchmark but used to drive only 2.78% of the
    # headline, because D4 was buried inside Reasoning and was near-constant on feasible tasks.
    # FOUR correctness categories in the conjunctive geomean; EFFICIENCY reported standalone.
    # Efficiency (d5_retry: tokens / tool-calls) is a resource-cost proxy, not a correctness
    # requirement — a correct itinerary produced expensively is still correct. Folding it into the
    # conjunctive headline asserts that being cheap is a precondition of being right, which is false,
    # and it let cost dominate a model's headline (deepseek efficiency 0.271 dragged an otherwise
    # strong planner down). It stays reported as `cat_efficiency` and the raw token/latency columns,
    # so the cost story is preserved as its own axis — just not multiplied into correctness.
    cats = {"cat_satisfaction": cat_satisfaction, "cat_truthfulness": cat_truthfulness,
            "cat_reasoning": cat_reasoning, "cat_infeasibility": agg["cat_infeasibility"]}
    missing = [k for k, v in cats.items() if v is None]
    agg["missing_categories"] = missing

    # GEOMETRIC MEAN, not a weighted sum. The five categories are CONJUNCTIVE requirements, not
    # tradeable features: a plan must be simultaneously constraint-satisfying, truthful, physically
    # executable, correctly refused when impossible, and affordable to produce. An arithmetic mean
    # asserts that perfect truthfulness compensates for an itinerary nobody can physically follow —
    # which is false, and it is what let a model that cannot schedule (planning 0.45) still score
    # 0.73 overall. The geometric mean lets no dimension hide behind the others; measured on the
    # 11-model pilot it preserved discrimination (sd 0.139 -> 0.137) while dropping the median from
    # 0.731 to 0.625 and correctly demoting a model whose efficiency was 0.047.
    # (Same aggregate the HDI uses, for the same non-substitutability reason.)
    if missing:
        agg["weighted_overall"] = None
        agg["arithmetic_overall"] = None
    else:
        vals = list(cats.values())
        prod = 1.0
        for v in vals:
            prod *= max(float(v), 1e-9)     # floor keeps a single 0 from erasing all information
        agg["weighted_overall"] = prod ** (1.0 / len(vals))
        agg["arithmetic_overall"] = sum(vals) / len(vals)   # reported for comparison
    
    # 保留原始简单平均 (用于对比)
    all_scores = [v for k, v in agg.items() if v is not None and k.startswith(("d0", "d1", "d2", "d3", "d4", "d5", "b2", "b3"))]
    agg["overall_avg"] = sum(all_scores) / len(all_scores) if all_scores else None
    
    return agg


def bootstrap_aggregate(results: list, n_boot: int = 2000, seed: int = 42,
                        keys=("weighted_overall", "cat_satisfaction", "cat_truthfulness",
                              "cat_reasoning", "cat_efficiency")) -> dict:
    """Percentile bootstrap CIs for the headline and the four categories, resampling QUERIES.

    The benchmark previously reported point estimates only, and ranked adjacent models on gaps of
    1.0-2.3 points — well inside the sampling noise (a paired bootstrap put the 95% CI of a
    1.82-point gap at [-1.64, +5.95]). Any claim that model A beats model B needs this.

    Deterministic: fixed seed, and the resampling draws indices with a seeded RNG, so the same
    inputs always yield the same interval.
    """
    import random as _random
    n = len(results)
    out = {k: {"mean": None, "lo": None, "hi": None} for k in keys}
    if n == 0:
        return out
    rng = _random.Random(seed)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        agg = compute_aggregate_scores(sample)
        for k in keys:
            v = agg.get(k)
            if v is not None:
                draws[k].append(v)
    base = compute_aggregate_scores(results)
    for k in keys:
        d = sorted(draws[k])
        if not d:
            continue
        out[k] = {"mean": base.get(k),
                  "lo": d[int(0.025 * (len(d) - 1))],
                  "hi": d[int(0.975 * (len(d) - 1))]}
    return out


def paired_bootstrap_diff(results_a: list, results_b: list, n_boot: int = 2000, seed: int = 42,
                          key: str = "weighted_overall") -> dict:
    """PAIRED bootstrap of (A - B) on the SAME queries — the right test for 'does A beat B'.

    Returns the observed gap, its 95% CI and whether the interval excludes 0. Pairing by
    query_index removes between-task variance, which is what makes a small but consistent
    advantage detectable at all.
    """
    import random as _random
    by_b = {r.query_index: r for r in results_b}
    pairs = [(a, by_b[a.query_index]) for a in results_a if a.query_index in by_b]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "diff": None, "lo": None, "hi": None, "significant": False}
    rng = _random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [pairs[i][0] for i in idx]
        sb = [pairs[i][1] for i in idx]
        va = compute_aggregate_scores(sa).get(key)
        vb = compute_aggregate_scores(sb).get(key)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    obs_a = compute_aggregate_scores([p[0] for p in pairs]).get(key)
    obs_b = compute_aggregate_scores([p[1] for p in pairs]).get(key)
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))] if diffs else None
    hi = diffs[int(0.975 * (len(diffs) - 1))] if diffs else None
    return {"n": n,
            "diff": (obs_a - obs_b) if (obs_a is not None and obs_b is not None) else None,
            "lo": lo, "hi": hi,
            "significant": bool(lo is not None and hi is not None and (lo > 0 or hi < 0))}


# ============ 包装函数用于并发 ============
def process_single_query(args):
    """
    单个查询处理函数 (用于并发)
    args: (index, result_data, meta, scorer)
    """
    i, result, meta, scorer = args
    
    query_idx = result.get("query_index", i)
    plan_output = result.get("plan", {})
    retry_count = result.get("tool_call_count", 0)
    
    try:
        score = scorer.score_query(plan_output, meta, retry_count)
        score.query_index = query_idx
        return score
    except Exception as e:
        print(f"Error scoring query {query_idx}: {e}")
        return None

# ============ 主程序 ============
def main():
    parser = argparse.ArgumentParser(description="KDD Travel Planning Scoring Script")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file (LLM pipeline output)")
    parser.add_argument("--meta", "-m", required=True, help="Query metadata CSV file")
    parser.add_argument("--output", "-o", help="Output scores CSV file")
    parser.add_argument("--beta", type=float, default=4.0, help="Budget decay coefficient (D3)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overflow decay coefficient (B1)")
    parser.add_argument("--k-threshold", type=int, default=15, help="Attraction threshold (B1)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    # 引入进度条
    try:
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
    except ImportError:
        print("Please install tqdm: pip install tqdm")
        sys.exit(1)

    # 加载数据
    print(f"Loading LLM outputs from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        llm_data = json.load(f)
    
    results_list = llm_data.get("results", [])
    print(f"Loaded {len(results_list)} results")
    
    print(f"Loading query metadata from {args.meta}...")
    queries = load_queries(args.meta)
    print(f"Loaded {len(queries)} queries")
    
    # 创建评分器
    db = get_sandbox_db()
    
    # D1 is deterministic rule-based facility matching (no embedding, no threshold); nothing to
    # preload. The old semantic path and its Bedrock-Titan preload are retired.
    
    scorer = TravelPlanScorer(db, {
        "beta": args.beta,
        "alpha": args.alpha,
        "k_threshold": args.k_threshold
    })
    
    # 准备任务
    print(f"\nScoring with {args.workers} threads...")
    tasks = []
    for i, result in enumerate(results_list):
        query_idx = result.get("query_index", i)
        if query_idx >= len(queries):
            print(f"Warning: query_index {query_idx} out of range, skipping")
            continue
        
        meta = queries[query_idx]
        tasks.append((i, result, meta, scorer))
    
    # 并发执行
    score_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 使用 tqdm 显示进度
        results = list(tqdm(executor.map(process_single_query, tasks), total=len(tasks), unit="q"))
        
        # 过滤失败结果
        score_results = [r for r in results if r is not None]
    
    print(f"Successfully scored {len(score_results)} queries")
    
    # 汇总统计
    agg = compute_aggregate_scores(score_results)
    
    print("\n" + "=" * 50)
    print("AGGREGATE SCORES Code Updated")
    print("=" * 50)
    for key, value in agg.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: N/A")
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"scores_{input_name}_{timestamp}.csv"
    
    # 转为 DataFrame
    df = pd.DataFrame([asdict(r) for r in score_results])
    df.to_csv(output_path, index=False)
    print(f"\nScores saved to {output_path}")
    
    # 保存汇总 JSON
    agg_path = output_path.replace(".csv", "_aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "beta": args.beta,
                "alpha": args.alpha,
                "k_threshold": args.k_threshold
            },
            "input_file": args.input,
            "meta_file": args.meta,
            "total_queries": len(score_results),
            "aggregate_scores": agg
        }, f, ensure_ascii=False, indent=2)
    print(f"Aggregate scores saved to {agg_path}")


if __name__ == "__main__":
    main()
