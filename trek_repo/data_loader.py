"""
KDD Travel Planning - Data Loader Module
=========================================
提供查询元数据加载和沙箱数据库访问功能。
"""

import os
import ast
import re
import pandas as pd
from dataclasses import dataclass, field
import unicodedata
import glob
from typing import Optional, Any


# ============ 路径配置 ============
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(CUR_DIR, "api")
# v2 is the cleaned KB and the only one evaluated against; v1 is kept read-only as build input.
DATA_DIR = os.path.join(API_DIR, "data", "v2")

FLIGHT_CSV_PATH = os.path.join(DATA_DIR, "flight_data", "flights.csv")


# ----------------------------------------------------------------------------
# City resolution
# ----------------------------------------------------------------------------
# This module used to reach a per-city file by interpolating the requested city straight into a
# filename. That silently missed 37 of 375 cities -- v1 escaped apostrophes to "_" ("Xi'an" ->
# Xi_an_rental_cars.csv) and stored 27 names NFD while the city_name column is NFC. The build now
# names each file for its NFC city_name, and lookups resolve through an index of what is on disk,
# folded case- and accent-insensitively, so a miss means the KB really has no such city.
def _fold(name: str) -> str:
    s = unicodedata.normalize('NFKD', str(name)).strip().lower()
    return ''.join(c for c in s if not unicodedata.combining(c))


_CITY_INDEX = {}


def _resolve_city_path(city: str, directory: str, suffix: str):
    """Resolve a city to its KB file. EXACT filename first, diacritic-folded only as a fallback.

    'San Jose' (United States, 37.3N) and 'San José' (Costa Rica, 10.0N) are two different cities
    in the KB but fold to the same key, and the folded index kept only whichever glob returned
    first -- which differed per domain, so one city's hotels were served with the other's
    attractions, 4,837 km apart. Exact match first keeps them distinct; folding still covers
    NFC/NFD and case differences for every other city.
    """
    key = (directory, suffix)
    if key not in _CITY_INDEX:
        exact, folded = {}, {}
        # SORTED: glob order is filesystem-dependent, so which member of a folded collision
        # (San Jose / San José) claimed the folded slot changed between machines and made scores
        # irreproducible. Sorting makes the winner the lexicographically first name, always.
        for path in sorted(glob.glob(os.path.join(directory, '*' + suffix))):
            base = os.path.basename(path)[:-len(suffix)]
            exact[unicodedata.normalize('NFC', base)] = path
            folded.setdefault(_fold(base), path)
        _CITY_INDEX[key] = (exact, folded)
    exact, folded = _CITY_INDEX[key]
    # NFC-normalise the lookup too: 27 KB filenames were stored NFD, so a raw string compare
    # against an NFC city name missed and silently fell through to the folded index.
    return exact.get(unicodedata.normalize('NFC', str(city))) or folded.get(_fold(city))
HOTEL_DATA_DIR = os.path.join(DATA_DIR, "hotel_data")
ATTRACTION_DATA_DIR = os.path.join(DATA_DIR, "attraction_data")
CAR_DATA_DIR = os.path.join(DATA_DIR, "car_data")


# ============ 查询元数据 ============
@dataclass
class QueryMeta:
    """查询元数据，解析自 query CSV"""
    query: str
    level: str  # easy/medium/hard
    person_num: int = 1
    budget: float = 0.0
    days: int = 3
    rooms_count: int = 1
    implicit_keywords: list[str] = field(default_factory=list)
    hard_constraints: dict = field(default_factory=dict)
    req_flight: dict = field(default_factory=dict)
    req_hotel: dict = field(default_factory=dict)
    req_car: dict = field(default_factory=dict)
    req_attraction: dict = field(default_factory=dict)
    cities_count: int = 1
    is_ordered: bool = False
    impossible: bool = False
    
    @classmethod
    def from_csv_row(cls, row: pd.Series) -> "QueryMeta":
        """从 CSV 行解析元数据"""
        def safe_eval(val, default=None):
            if pd.isna(val) or val == "" or val == "nan":
                return default
            try:
                return ast.literal_eval(str(val))
            except:
                return default
        
        # 解析 hard_constraints
        hard_constraints = safe_eval(row.get("hard_constraints"), {})
        
        # 解析 implicit_keywords
        implicit_kw_raw = safe_eval(row.get("implicit_keywords"), [])
        implicit_keywords = implicit_kw_raw if isinstance(implicit_kw_raw, list) else []
        
        # 解析航班需求
        req_flight = safe_eval(row.get("req_flight"), {})
        
        # 解析酒店需求
        req_hotel = safe_eval(row.get("req_hotel"), {})
        
        # 解析租车需求
        req_car = safe_eval(row.get("req_car"), {})
        
        # 解析景点需求
        req_attraction = safe_eval(row.get("req_attraction"), {})
        
        # 解析城市数量
        cities_count_raw = safe_eval(row.get("cities_count"), {"cities_count": 1})
        cities_count = cities_count_raw.get("cities_count", 1) if isinstance(cities_count_raw, dict) else 1
        
        # 解析 impossible 标记
        impossible = str(row.get("impossible", "")).lower() in ("true", "1", "yes", "1.0")
        
        # 类型转换辅助函数
        def to_int(val, default=0):
            try:
                return int(val)
            except (ValueError, TypeError):
                return default
        
        def to_float(val, default=0.0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # 判断是否有顺序要求
        # 检查查询文本中是否包含顺序关键词
        query_text = str(row.get("query", "")).lower()
        is_ordered = any(keyword in query_text for keyword in [
            "first", "then", "after", "before", "next", "finally",
            "day 1", "day 2", "day 3", "day1", "day2", "day3",
            "先", "然后", "接着", "之后", "最后"  # 中文顺序词
        ])
        
        # 如果航班有多个目的地且查询中包含顺序词，则认为是有序的
        # 如果仅仅是多城市且包含顺序词，则认为是 d2_ord_multi
        # 之前的逻辑过于依赖 req_flight 解析，这里放宽为 cities_count 判断
        if cities_count > 1 and is_ordered:
            is_ordered = True
        else:
            is_ordered = False
        
        return cls(
            query=str(row.get("query", "")),
            level=str(row.get("level", "easy")),
            person_num=to_int(hard_constraints.get("person_num", 1), 1),
            budget=to_float(hard_constraints.get("budget", 0), 0),
            days=to_int(hard_constraints.get("days", 3), 3),
            rooms_count=to_int(req_hotel.get("rooms_count", 1), 1) if req_hotel else 1,
            implicit_keywords=implicit_keywords,
            hard_constraints=hard_constraints,
            req_flight=req_flight,
            req_hotel=req_hotel,
            req_car=req_car,
            req_attraction=req_attraction,
            cities_count=cities_count,
            is_ordered=is_ordered,
            impossible=impossible,
        )


def load_queries(csv_path: str) -> list[QueryMeta]:
    """加载查询 CSV 文件"""
    df = pd.read_csv(csv_path)
    return [QueryMeta.from_csv_row(row) for _, row in df.iterrows()]


# ============ 沙箱数据库 ============
class SandboxDB:
    """沙箱数据库封装，用于验证 LLM 输出"""
    
    def __init__(self):
        self._flights_df: Optional[pd.DataFrame] = None
        self._hotels_cache: dict[str, pd.DataFrame] = {}
        self._attractions_cache: dict[str, pd.DataFrame] = {}
        self._cars_cache: dict[str, pd.DataFrame] = {}
    
    @property
    def flights(self) -> pd.DataFrame:
        """懒加载航班数据"""
        if self._flights_df is None:
            if os.path.exists(FLIGHT_CSV_PATH):
                self._flights_df = pd.read_csv(FLIGHT_CSV_PATH)
            else:
                self._flights_df = pd.DataFrame()
        return self._flights_df
    
    def get_hotels(self, city: str) -> pd.DataFrame:
        """获取城市酒店数据"""
        if city not in self._hotels_cache:
            csv_path = _resolve_city_path(city, HOTEL_DATA_DIR, "_hotel.csv")
            if csv_path:
                self._hotels_cache[city] = pd.read_csv(csv_path)
            else:
                self._hotels_cache[city] = pd.DataFrame()
        return self._hotels_cache[city]
    
    def get_attractions(self, city: str) -> pd.DataFrame:
        """获取城市景点数据"""
        if city not in self._attractions_cache:
            csv_path = _resolve_city_path(city, ATTRACTION_DATA_DIR, "_attraction.csv")
            if csv_path:
                self._attractions_cache[city] = pd.read_csv(csv_path)
            else:
                self._attractions_cache[city] = pd.DataFrame()
        return self._attractions_cache[city]
    
    def get_cars(self, city: str) -> pd.DataFrame:
        """获取城市租车数据"""
        if city not in self._cars_cache:
            csv_path = _resolve_city_path(city, CAR_DATA_DIR, "_rental_cars.csv")
            if csv_path:
                self._cars_cache[city] = pd.read_csv(csv_path)
            else:
                self._cars_cache[city] = pd.DataFrame()
        return self._cars_cache[city]
    
    def verify_flight(self, flight_number: str, departure_city: str = None, 
                      arrival_city: str = None, price: float = None) -> bool:
        """验证航班是否存在且数据一致"""
        df = self.flights
        if df.empty:
            return False
        
        # case/whitespace-insensitive, matching verify_hotel/verify_attraction — an exact match here
        # (while hotels/attractions fold) turned harmless casing drift into a false hallucination
        mask = df["flight_number"].astype(str).str.strip().str.lower() == str(flight_number).strip().lower()
        if departure_city:
            mask &= df["departure_city"].astype(str).str.strip().str.lower() == str(departure_city).strip().lower()
        if arrival_city:
            mask &= df["arrival_city"].astype(str).str.strip().str.lower() == str(arrival_city).strip().lower()
        
        matches = df[mask]
        if matches.empty:
            return False
        
        if price is not None:
            # 允许小数误差
            return any(abs(matches["price"] - price) < 0.01)
        return True
    
    def flight_times_match(self, flight_number: str, departure_time=None, arrival_time=None) -> bool:
        """True iff the STATED departure/arrival times equal the KB flight's own times.

        B3 measures reachability against the plan's *stated* flight times and B2 only checks
        attraction hours, so a model could state arbitrary flight times and pass every check. This
        closes that hole: a real flight number carries fixed times, and claiming different ones is a
        fabricated schedule. Verified against the SAME KB the scorer loads (data/v2); the stale
        top-level data/flight_data would show 28 false mismatches on gold. Times not supplied are not
        checked (absence is B3's business, not truthfulness's).
        """
        df = self.flights
        if df.empty:
            return True
        m = df[df["flight_number"].astype(str).str.strip().str.lower()
               == str(flight_number).strip().lower()]
        if m.empty:
            return True   # existence is verify_flight's job; don't double-penalise here
        def _norm(t):
            return str(t).strip() if t not in (None, "") else None
        for col, stated in (("departure_time", departure_time), ("arrival_time", arrival_time)):
            st = _norm(stated)
            if st is None:
                continue
            kb_vals = {_norm(v) for v in m[col].tolist()}
            if st not in kb_vals:
                return False
        return True

    def verify_hotel(self, name: str, city: str, price: float = None) -> bool:
        """验证酒店是否存在"""
        df = self.get_hotels(city)
        if df.empty:
            return False
        
        matches = df[df["name"].str.lower() == name.lower()]
        if matches.empty:
            return False
        
        if price is not None:
            return any(abs(matches["price"] - price) < 0.01)
        return True
    
    def verify_attraction(self, name: str, city: str) -> bool:
        """验证景点是否存在"""
        df = self.get_attractions(city)
        if df.empty:
            return False

        return any(df["attraction_name"].str.lower() == name.lower())

    def verify_car(self, car_type: str, city: str) -> bool:
        """验证该城市是否存在此车型 (D0-src/D4 此前根本不核实车 — 240 个租车任务的真实性豁免)。"""
        if not car_type:
            return False
        df = self.get_cars(city)
        if df.empty:
            return False
        return any(df["car_type"].astype(str).str.strip().str.lower() == str(car_type).strip().lower())
    
    def get_open_hours(self, attraction_name: str, city: str) -> Optional[str]:
        """获取景点营业时间"""
        df = self.get_attractions(city)
        if df.empty:
            return None

        matches = df[df["attraction_name"].str.lower() == attraction_name.lower()]
        if matches.empty:
            return None

        return str(matches.iloc[0].get("open_hours", ""))

    def get_visit_duration(self, attraction_name: str, city: str) -> Optional[float]:
        """Hours the KB says this attraction takes, or None if unknown.

        B2 needs it to check that a scheduled visit actually leaves enough time for the visit —
        the agent prompt demands `visit_end - visit_start` cover `duration_of_visit`, so the
        scorer has to verify it rather than take the agent's word.

        The KB stores this as prose ("2 hours", "3 hours") for all 55,814 rows, so the old
        `float(val)` raised ValueError on every single one and silently returned None. B2's
        minimum-visit-duration check has therefore never executed once, and a one-minute stop at a
        three-hour museum scored as fully compliant. Parse the leading number instead.
        """
        df = self.get_attractions(city)
        if df.empty:
            return None
        matches = df[df["attraction_name"].astype(str).str.lower() == str(attraction_name).lower()]
        if matches.empty:
            return None
        val = matches.iloc[0].get("duration_of_visit")
        if pd.isna(val):
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", str(val))
        return float(m.group(1)) if m else None
    
    def get_coordinates(self, entity_name: str, entity_type: str, city: str) -> Optional[tuple[float, float]]:
        """获取实体坐标 (latitude, longitude)"""
        if entity_type == "hotel":
            df = self.get_hotels(city)
            name_col = "name"
        elif entity_type == "attraction":
            df = self.get_attractions(city)
            name_col = "attraction_name"
        elif entity_type == "flight":
            # A flight occupies TWO places; this legacy entry point returns where it ENDS.
            # Callers that need the boarding point must use get_flight_endpoint(fn, "departure").
            return self.get_flight_endpoint(entity_name, "arrival")
        else:
            return None
        
        if df.empty:
            return None
        
        matches = df[df[name_col].str.lower() == entity_name.lower()]
        if matches.empty:
            return None
        
        row = matches.iloc[0]
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            return None
        return (float(lat), float(lon))
    
    def get_flight_endpoint(self, flight_number: str, which: str = "arrival") -> Optional[tuple]:
        """Coordinates of a flight's DEPARTURE or ARRIVAL airport.

        A flight is the only event that starts and ends in different places: you board at the
        departure airport and land at the arrival one. B3 must measure a transition INTO a flight
        against the departure airport and a transition OUT of one against the arrival airport —
        using the arrival airport for both (the previous behaviour) let a plan leave an attraction
        2,663 km from its boarding gate 95 minutes before take-off and score perfectly, while
        penalising legitimate drives to the airport.

        Matching is case/whitespace-insensitive, the same rule verify_flight uses.
        """
        df = self.flights
        if df.empty:
            return None
        m = (df["flight_number"].astype(str).str.strip().str.lower()
             == str(flight_number).strip().lower())
        sub = df[m]
        if sub.empty:
            return None
        row = sub.iloc[0]
        pre = "departure" if str(which).lower().startswith("dep") else "arrival"
        lat, lon = row.get(f"{pre}_airport_latitude"), row.get(f"{pre}_airport_longitude")
        if pd.isna(lat) or pd.isna(lon):
            return None
        try:
            return (float(lat), float(lon))
        except (TypeError, ValueError):
            return None

    def get_flight_price(self, flight_number: str) -> Optional[float]:
        """从 DB 获取航班价格"""
        df = self.flights
        if df.empty:
            return None
        
        matches = df[df["flight_number"] == flight_number]
        if matches.empty:
            return None
        
        price = matches.iloc[0].get("price")
        return float(price) if pd.notna(price) else None
    
    def cheapest_flight_price(self, dep_city: str, arr_city: str) -> Optional[float]:
        """Cheapest direct fare for a required leg, or None if no direct flight exists.

        Used as the imputed floor for a REQUIRED leg the plan omitted, so dropping a leg cannot
        buy budget compliance. Mirrors the generator's cheapest-direct-flight term in C_floor.
        """
        if not dep_city or not arr_city:
            return None
        if not hasattr(self, "_cheapest_flight_cache"):
            self._cheapest_flight_cache = {}
        key = (str(dep_city).strip().lower(), str(arr_city).strip().lower())
        if key in self._cheapest_flight_cache:
            return self._cheapest_flight_cache[key]
        df = self.flights
        val = None
        if not df.empty:
            m = ((df["departure_city"].astype(str).str.strip().str.lower() == key[0]) &
                 (df["arrival_city"].astype(str).str.strip().str.lower() == key[1]))
            sub = df[m]
            if not sub.empty:
                prices = pd.to_numeric(sub["price"], errors="coerce").dropna()
                if not prices.empty:
                    val = float(prices.min())
        self._cheapest_flight_cache[key] = val
        return val

    def cheapest_car_price(self, city: str, min_capacity=None) -> Optional[float]:
        """Cheapest daily rate in `city` (optionally meeting a capacity floor), or None."""
        if not city:
            return None
        if not hasattr(self, "_cheapest_car_cache"):
            self._cheapest_car_cache = {}
        key = (str(city), int(min_capacity) if min_capacity else 0)
        if key in self._cheapest_car_cache:
            return self._cheapest_car_cache[key]
        df = self.get_cars(city)
        val = None
        if not df.empty and "price_per_day" in df.columns:
            sub = df
            if min_capacity and "capacity" in df.columns:
                cap = pd.to_numeric(df["capacity"], errors="coerce")
                sub = df[cap >= int(min_capacity)]
            prices = pd.to_numeric(sub["price_per_day"], errors="coerce").dropna() if not sub.empty else None
            if prices is not None and not prices.empty:
                val = float(prices.min())
        self._cheapest_car_cache[key] = val
        return val

    def cheapest_hotel_price(self, city: str) -> Optional[float]:
        """Cheapest nightly rate available in `city`, or None if the city has no hotels.

        This is the imputed floor D3 charges for a stay-city night the plan booked no hotel for.
        It is the SAME quantity the generator's C_floor uses, which is what keeps the
        budget-infeasible labels true w.r.t. the scorer's own bill: an agent cannot get under an
        'impossible' budget simply by omitting lodging.
        """
        if not city:
            return None
        if not hasattr(self, "_cheapest_hotel_cache"):
            self._cheapest_hotel_cache = {}
        key = str(city)
        if key in self._cheapest_hotel_cache:
            return self._cheapest_hotel_cache[key]
        df = self.get_hotels(city)
        val = None
        if not df.empty and "price" in df.columns:
            prices = pd.to_numeric(df["price"], errors="coerce").dropna()
            if not prices.empty:
                val = float(prices.min())
        self._cheapest_hotel_cache[key] = val
        return val

    def get_hotel_price(self, name: str, city: str = None) -> Optional[float]:
        """从 DB 获取酒店价格"""
        # 尝试在指定城市查找
        if city:
            df = self.get_hotels(city)
            if not df.empty:
                matches = df[df["name"].str.lower() == name.lower()]
                if not matches.empty:
                    price = matches.iloc[0].get("price")
                    return float(price) if pd.notna(price) else None

        # A city was asked for and has no such hotel: that is an absent price, not another
        # city's. Falling through to the global scan below would charge D3 the first same-named
        # hotel in os.listdir order -- 'green valley garden resort' spans 17 cities at 17.0 to
        # 1202.0. Same fix already applied to get_car_price.
        if city:
            return None

        # 没有城市时才遍历所有城市
        for csv_file in sorted(os.listdir(HOTEL_DATA_DIR)):
            if csv_file.endswith("_hotel.csv"):
                city_name = csv_file.replace("_hotel.csv", "")
                df = self.get_hotels(city_name)
                if not df.empty:
                    matches = df[df["name"].str.lower() == name.lower()]
                    if not matches.empty:
                        price = matches.iloc[0].get("price")
                        return float(price) if pd.notna(price) else None

        return None
    
    def get_attraction_price(self, name: str, city: str = None) -> Optional[float]:
        """从 DB 获取景点门票价格"""
        # 尝试在指定城市查找
        if city:
            df = self.get_attractions(city)
            if not df.empty:
                matches = df[df["attraction_name"].str.lower() == name.lower()]
                if not matches.empty:
                    price = matches.iloc[0].get("ticket_price")
                    return float(price) if pd.notna(price) else None

        # Same guard as hotels/cars: a named city that misses is an absent price, not another
        # city's attraction of the same name.
        if city:
            return None

        # 没有城市时才遍历所有城市
        for csv_file in sorted(os.listdir(ATTRACTION_DATA_DIR)):
            if csv_file.endswith("_attraction.csv"):
                city_name = csv_file.replace("_attraction.csv", "")
                df = self.get_attractions(city_name)
                if not df.empty:
                    matches = df[df["attraction_name"].str.lower() == name.lower()]
                    if not matches.empty:
                        price = matches.iloc[0].get("ticket_price")
                        return float(price) if pd.notna(price) else None

        return None
    
    def get_car_price(self, car_type: str = None, city: str = None,
                      claimed: float = None) -> Optional[float]:
        """The KB price of the car a plan booked.

        Every (city, car_type) holds three cars at three different prices -- they differ by their
        extra_services, which is what D1/D2 reads. `car_type` alone therefore does not identify a
        car, and this used to answer with `iloc[0]`, i.e. whichever of the three the file happened
        to list first. That charged D3 for a car the agent never booked: Beijing/Minivan is
        139.88, 98.15, 106.49, so a plan booking the 98.15 one was billed 139.88 -- +375.57 over
        nine days, and a perfect plan scored D3 = 0.798. It was not a correctable bias either;
        `iloc[0]` ranks 2.0 of 3 on average, which is exactly chance. 115 of 173 car-booking tasks
        were priced this way.

        A plan already names its price, so `claimed` identifies which of the three it means. If
        the city charges that price for that car type, that is the car, and its price is used.
        If no car matches the claim, the plan is not describing a real booking -- that is D0-src's
        to penalise, not D3's to reward -- so the nearest real price is charged instead. Claiming
        $1 gets billed the cheapest car that exists, never $1.
        """
        if city:
            df = self.get_cars(city)
            if not df.empty:
                if car_type:
                    matches = df[df["car_type"].str.lower() == car_type.lower()]
                else:
                    matches = df
                if not matches.empty:
                    prices = pd.to_numeric(matches["price_per_day"], errors="coerce").dropna()
                    if prices.empty:
                        return None
                    if claimed is not None:
                        try:
                            want = float(claimed)
                        except (TypeError, ValueError):
                            want = None
                        if want is not None:
                            return float(prices.iloc[(prices - want).abs().argmin()])
                    return float(prices.iloc[0])
        
        # A city was asked for and the KB has no such city, or no such car in it. Previously this
        # fell through to a loop over every city that ignored the `city` argument and returned the
        # first matching car found in os.listdir order -- so get_car_price('EV', "Xi'an") answered
        # 116.14, which is Amsterdam's price. Only fall back to a global scan when no city was
        # given at all; a city that was asked for and missed is an absent price, not another
        # continent's.
        if city:
            return None

        for csv_file in sorted(os.listdir(CAR_DATA_DIR)):
            if csv_file.endswith("_rental_cars.csv"):
                city_name = csv_file.replace("_rental_cars.csv", "")
                df = self.get_cars(city_name)
                if not df.empty:
                    if car_type:
                        matches = df[df["car_type"].str.lower() == car_type.lower()]
                    else:
                        matches = df
                    if not matches.empty:
                        price = matches.iloc[0].get("price_per_day")
                        return float(price) if pd.notna(price) else None

        return None


# ============ 全局实例 ============
_sandbox_db: Optional[SandboxDB] = None

def get_sandbox_db() -> SandboxDB:
    """获取沙箱数据库单例"""
    global _sandbox_db
    if _sandbox_db is None:
        _sandbox_db = SandboxDB()
    return _sandbox_db


if __name__ == "__main__":
    # 测试加载
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        queries = load_queries(csv_path)
        print(f"Loaded {len(queries)} queries from {csv_path}")
        if queries:
            q = queries[0]
            print(f"First query: {q.query[:80]}...")
            print(f"  Level: {q.level}, Budget: {q.budget}, Days: {q.days}")
            print(f"  Implicit keywords: {q.implicit_keywords}")
    else:
        # 测试数据库
        db = get_sandbox_db()
        print(f"Flights loaded: {len(db.flights)} rows")
