"""
KDD Travel Planning - Implicit Requirements Scoring Module (D1)
================================================================
基于隐式需求详细配置表的结构化评分逻辑。

评分规则（v4 - 确定性关键词匹配）：
- 设施匹配：_facility_hit 做确定性大小写/子串匹配（persona 关键词表见本文件顶部三张表）
- 无 LLM、无 embedding、无相似度阈值：同一提交永远得同一分，完全可离线复现
- 双维度特例：luxury(酒店 star)、foodie(景点设施)
- 无要求：跳过检查，不计入分母

更新说明：
v4: 评分改回确定性关键词匹配。下方 check_facility_match_semantic / get_embedding_model /
    SEMANTIC_SIMILARITY_THRESHOLD 为历史死代码，活路径（score_hotel/attraction/car）只调 _facility_hit，
    评分器无需任何嵌入模型或凭据即可运行。
"""

import os
import ast
import json
import unicodedata
import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass

# 语义相似度支持 — Titan v2 via Bedrock (same model as the API's core_api, so query and stored
# vectors share one embedding space). Old local Qwen model retired.
import sys

# Resolve KB files through the SAME deterministic index the scorer uses. Interpolating the
# city straight into a filename only worked because macOS is case-insensitive: on a
# case-sensitive filesystem the lookup missed and D1 collapsed from 1.0 to 0.0, which broke
# the benchmark's reproducibility claim outright.
from data_loader import _resolve_city_path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "api"))
try:
    from bedrock_embed import embedding_model as _bedrock_encoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# ============ 路径配置 ============
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(CUR_DIR, "api")
# v2 is the cleaned KB and the only one evaluated against; v1 is kept read-only as build input.
DATA_DIR = os.path.join(API_DIR, "data", "v2")
CONFIG_DIR = os.path.join(CUR_DIR, "new_travelbench")

HOTEL_DATA_DIR = os.path.join(DATA_DIR, "hotel_data")
ATTRACTION_DATA_DIR = os.path.join(DATA_DIR, "attraction_data")
CAR_DATA_DIR = os.path.join(DATA_DIR, "car_data")

# ============ 历史死代码配置（活路径不使用）============
# D1 是确定性关键词匹配，不用相似度阈值。此常量仅为下方历史 semantic 函数保留，评分不依赖它。
SEMANTIC_SIMILARITY_THRESHOLD = 0.65  # DEAD: retained only for the unused semantic path below
EMBEDDING_MODEL_NAME = "amazon.titan-embed-text-v2:0"  # Bedrock; same model as core_api

# 全局嵌入模型实例（延迟加载）
_embedding_model = None

def get_embedding_model():
    """获取嵌入模型单例（Bedrock Titan v2，与 core_api 同一模型/向量空间）"""
    global _embedding_model
    if _embedding_model is None:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("bedrock_embed (Titan v2) unavailable; check AWS credentials in env")
        _embedding_model = _bedrock_encoder
    return _embedding_model


# ============ 隐式需求配置 ============
# 酒店设施关键词
HOTEL_FACILITY_KEYWORDS = {
    "with children": ["Child-friendly rooms", "Baby cots", "Stroller storage", "Diaper changing tables", "Nursing rooms"],
    "road trip": ["Parking", "24-hour front desk", "EV charging for electric vehicles"],
    "elderly travelers": ["Elevators", "Bathroom grab bars", "Medical contacts", "Breakfast with dietary options"],
    "business travelers": ["High speed WiFi", "Business center", "Meeting rooms", "Laundry service", "Print machine"],
    "with pets": ["Pet-friendly", "Pet rest area", "Pet beds", "Durable floors"],
    "nightlife enthusiast": ["24-hour front desk", "Late room service", "On-site bars", "Rooftop lounges/nightclubs"],
    "disabled traveler": ["Accessible entrances", "Ramps/lifts", "Accessible rooms/bathrooms", "Braille/raised signage"],
    "fast-paced budget travel": ["Free Wi-Fi", "Communal kitchens", "Laundry", "Lockers"],
    "couples trip": ["Bathtubs/jacuzzis", "Scenic rooms/villas", "Spa"],
    "solo women": ["Women-only floors/rooms", "Privacy-conscious check-in", "24-hour security", "Surveillance cameras", "Double locks on doors"],
    "luxury travelers": ["Spa", "Gym", "Premium bedding", "Bar"],  # 特殊：双维度
    "foodie": [],  # 特殊：仅评分维度
    "photography": ["Scenic rooms", "Sunrise calls", "Photography services/packages"],
}

# 景点设施关键词
ATTRACTION_FACILITY_KEYWORDS = {
    "with children": ["family restrooms", "Nursing facilities"],
    "road trip": ["parking"],
    "elderly travelers": ["wheelchair rental", "benches/rest areas"],
    "business travelers": ["High speed WiFi"],
    "with pets": ["Pet-friendly", "pet water stations", "Pet rest area"],
    "nightlife enthusiast": ["night markets", "bars", "evening shows"],
    "disabled traveler": ["ramps", "elevators", "wheelchair rentals", "accessible restrooms"],
    "fast-paced budget travel": ["city passes", "luggage storage", "self-guided tours"],
    "couples trip": ["scenic spots", "sunset cruises", "special couples experiences"],
    "solo women": ["group tours", "security"],
    "luxury travelers": ["private tours", "skip the line passes", "VIP exclusive events"],
    "foodie": ["Food markets", "tasting tours"],  # 特殊：双维度
    "photography": ["photo spots", "guided photo tours", "charging stations"],
}

# 租车设施关键词
CAR_FACILITY_KEYWORDS = {
    "with children": ["Child seats", "Child locks"],
    "road trip": ["Roadside emergency support"],
    "elderly travelers": ["Advanced Driving Assistance Systems"],
    "business travelers": ["WiFi"],
    "with pets": ["Pet seat belts", "Pet-friendly"],
    "nightlife enthusiast": ["Late pick up services"],
    "disabled traveler": ["Wheelchair space", "Hand controls", "Accessible cars"],
    "fast-paced budget travel": [],  # 跳过检查
    "couples trip": [],  # 跳过检查
    "solo women": ["Women-only cars", "Airport safe waiting areas"],
    "luxury travelers": ["First-class car", "Chauffeured cars", "Guide/driver package"],
    "foodie": [],  # 跳过检查
    "photography": ["Remote shoots"],
}

# 特殊规则配置
SPECIAL_RULES = {
    "luxury travelers": {
        "hotel": {"dual_dimension": True, "star_threshold": 5}
    },
    "foodie": {
        "hotel": {"rating_only": True, "restaurant_rating_threshold": 4.0},
        "attraction": {"dual_dimension": True, "restaurant_rating_threshold": 3.5}
    }
}


# ============ 数据库缓存 ============
class ImplicitScoringDB:
    """隐式需求评分专用数据库"""
    
    def __init__(self):
        self._hotels_cache: dict[str, pd.DataFrame] = {}
        self._attractions_cache: dict[str, pd.DataFrame] = {}
        self._cars_cache: dict[str, pd.DataFrame] = {}
    
    def get_hotels(self, city: str) -> pd.DataFrame:
        if city not in self._hotels_cache:
            csv_path = _resolve_city_path(city, HOTEL_DATA_DIR, "_hotel.csv")
            if csv_path:
                self._hotels_cache[city] = pd.read_csv(csv_path)
            else:
                self._hotels_cache[city] = pd.DataFrame()
        return self._hotels_cache[city]
    
    def get_attractions(self, city: str) -> pd.DataFrame:
        if city not in self._attractions_cache:
            csv_path = _resolve_city_path(city, ATTRACTION_DATA_DIR, "_attraction.csv")
            if csv_path:
                self._attractions_cache[city] = pd.read_csv(csv_path)
            else:
                self._attractions_cache[city] = pd.DataFrame()
        return self._attractions_cache[city]
    
    def get_cars(self, city: str) -> pd.DataFrame:
        if city not in self._cars_cache:
            csv_path = _resolve_city_path(city, CAR_DATA_DIR, "_rental_cars.csv")
            if csv_path:
                self._cars_cache[city] = pd.read_csv(csv_path)
            else:
                self._cars_cache[city] = pd.DataFrame()
        return self._cars_cache[city]
    
    def get_hotel_info(self, name: str, city: str) -> Optional[dict]:
        """获取酒店详细信息"""
        df = self.get_hotels(city)
        if df.empty:
            return None
        matches = df[df["name"].str.lower() == name.lower()]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {
            "name": row.get("name"),
            "amenities": self._parse_list(row.get("amenities", "[]")),
            "star": float(row.get("star", 0)),
            "rating": float(row.get("rating", 0)),
            "restaurant_rating": float(row.get("rate_of_restaurant", 0)),
        }
    
    def get_attraction_info(self, name: str, city: str) -> Optional[dict]:
        """获取景点详细信息"""
        df = self.get_attractions(city)
        if df.empty:
            return None
        matches = df[df["attraction_name"].str.lower() == name.lower()]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {
            "name": row.get("attraction_name"),
            "facilities": self._parse_list(row.get("facilities", "[]")),
            "restaurant_rating": float(row.get("rate_of_restaurant", 0)) if pd.notna(row.get("rate_of_restaurant")) else 0,
        }
    
    def get_car_info(self, car_type: str, city: str, claimed_price=None) -> Optional[dict]:
        """获取租车详细信息。

        每个 (city, car_type) 有 3 辆 extra_services 不同的车。若给了 claimed_price, 用它消歧到
        实际预订的那辆 (与 get_car_price 同一逻辑); 否则取第一辆。不消歧会让 D1 读到另一辆车的
        设施, gold 的 road-trip/pet 等 persona 车因此被误判为 0。

        PRICE TIES. `car_type` is not a unique key, and 6 of the 3,375 (city, car_type) groups hold
        two cars at the IDENTICAL price with different extra_services. There, price cannot
        disambiguate either, and `argmin` silently returned whichever row came first — which zeroed
        gold task #212 (a business-traveller Compact in Palma de Mallorca: the generator picked the
        $52.59 car WITH WiFi, the scorer resolved to the $52.59 car WITHOUT it). Nothing the plan
        records — type, city, price — distinguishes them, so charging the plan for that is charging
        it for an ambiguity in OUR data model. We therefore union the services of every row tied at
        the best price match. For the OR-shaped `_facility_hit` test this is exactly equivalent to
        scoring the best of the indistinguishable candidates, and the union is confined to that tie
        set — it never reaches a row the plan's own fields could have ruled out."""
        df = self.get_cars(city)
        if df.empty:
            return None
        if not isinstance(car_type, str):
            return None   # a malformed booking (e.g. car_type is itself a dict) is not a real car
        matches = df[df["car_type"].str.lower() == car_type.lower()]
        if matches.empty:
            matches = df[df["car_type"].str.lower().str.contains(car_type.lower(), regex=False)]
        if matches.empty:
            return None
        rows = [matches.iloc[0]]
        if claimed_price is not None:
            prices = pd.to_numeric(matches["price_per_day"], errors="coerce")
            if prices.notna().any():
                try:
                    dist = (prices - float(claimed_price)).abs()
                    tied = matches[dist == dist.min()]
                    rows = [tied.iloc[i] for i in range(len(tied))]
                except (TypeError, ValueError):
                    pass
        services, seen = [], set()
        for r in rows:
            for s in self._parse_list(r.get("extra_services", "[]")):
                k = str(s).strip().lower()
                if k not in seen:
                    seen.add(k)
                    services.append(s)
        return {
            "car_type": rows[0].get("car_type"),
            "extra_services": services,
        }
    
    def _parse_list(self, val) -> list:
        """解析列表字段"""
        if pd.isna(val) or val == "":
            return []
        try:
            if isinstance(val, list):
                return val
            return ast.literal_eval(str(val))
        except:
            return []


# ============ 评分函数 ============
def _fold_city(name) -> str:
    """Case/accent-insensitive city key, matching data_loader._fold, so 'Xi'an' and 'Xi\u2019an'
    or NFC/NFD spellings bucket together instead of splitting one city into two D1 cells."""
    s = unicodedata.normalize('NFKD', str(name)).strip().lower()
    return ''.join(c for c in s if not unicodedata.combining(c))


class ImplicitRequirementsScorer:
    """隐式需求评分器 (v3 - 语义相似度版本)"""
    
    def __init__(self, db: ImplicitScoringDB = None, similarity_threshold: float = None):
        self.db = db or ImplicitScoringDB()
        self.similarity_threshold = similarity_threshold or SEMANTIC_SIMILARITY_THRESHOLD
        self._embedding_model = None
        self._embedding_cache = {}  # 缓存嵌入向量
    
    def _get_model(self):
        """延迟加载嵌入模型"""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入（带缓存）"""
        if text not in self._embedding_cache:
            model = self._get_model()
            self._embedding_cache[text] = model.encode([text])[0]
        return self._embedding_cache[text]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def check_facility_match_semantic(self, resource_facilities: list, keywords: list) -> float:
        """
        使用语义相似度检查设施是否匹配需求关键词
        
        Args:
            resource_facilities: 资源的实际设施列表
            keywords: 需求关键词列表
        
        Returns:
            0-100 的分数，表示匹配程度
        """
        if not keywords or not resource_facilities:
            return 0.0
        
        # 将设施列表组合成一个文本
        facility_text = ", ".join(resource_facilities)
        facility_emb = self._get_embedding(facility_text)
        
        # 将需求关键词组合成一个文本
        keyword_text = ", ".join(keywords)
        keyword_emb = self._get_embedding(keyword_text)
        
        # 计算语义相似度
        similarity = self._cosine_similarity(facility_emb, keyword_emb)
        
        # 使用阈值判断：相似度 >= 阈值则满分，否则按比例给分
        # 这样可以区分"明确匹配"和"部分匹配"
        if similarity >= self.similarity_threshold:
            score = 100.0
        else:
            # 低于阈值时，按比例给分（0 到阈值映射到 0-100）
            score = (similarity / self.similarity_threshold) * 100
        
        return min(100.0, max(0.0, score))
    
    def check_facility_match(self, resource_facilities: list, keywords: list) -> bool:
        """
        兼容旧接口：使用语义相似度判断是否匹配（返回 bool）
        内部调用 check_facility_match_semantic
        """
        if not keywords:
            return False
        score = self.check_facility_match_semantic(resource_facilities, keywords)
        return score > 0
    
    @staticmethod
    def _facility_hit(resource_facilities: list, keywords: list) -> bool:
        """Deterministic: does the resource carry ANY of the persona's required facilities?
        Case-insensitive set intersection over the KB's fixed facility vocabulary — no embeddings,
        so D1 is fully reproducible (the reproducibility we sell against LLM-judge competitors).
        The KB facility strings and these keyword lists were verified to match on 417/417 gold
        resources, so the group-tag the generator filters on and this intersection agree."""
        if not keywords or not resource_facilities:
            return False
        fac = {str(f).strip().lower() for f in resource_facilities}
        kw = {str(k).strip().lower() for k in keywords}
        return bool(fac & kw)

    def score_hotel(self, hotel_info: dict, implicit_keyword: str) -> Optional[float]:
        """
        评分单个酒店对单个隐式需求的命中情况 (v3 - 语义相似度版本)
        返回 None 表示跳过检查
        返回 0-100 的分数
        """
        kw_lower = implicit_keyword.lower()
        keywords = HOTEL_FACILITY_KEYWORDS.get(kw_lower, [])
        special = SPECIAL_RULES.get(kw_lower, {}).get("hotel", {})

        amenities = hotel_info.get("amenities", [])
        star = hotel_info.get("star", 0)
        restaurant_rating = hotel_info.get("restaurant_rating", 0)

        # luxury: dual dimension — amenity present AND star >= 5, half credit each (deterministic)
        if special.get("dual_dimension"):
            facility_score = 100.0 if self._facility_hit(amenities, keywords) else 0.0
            star_score = 100.0 if star >= special.get("star_threshold", 5) else 0.0
            return (facility_score + star_score) / 2

        # foodie: restaurant rating only
        if special.get("rating_only"):
            return 100.0 if restaurant_rating >= special.get("restaurant_rating_threshold", 4.0) else 0.0

        if not keywords:
            return None  # this persona places no requirement on hotels -> skip
        return 100.0 if self._facility_hit(amenities, keywords) else 0.0
    
    def score_attraction(self, attraction_info: dict, implicit_keyword: str) -> Optional[float]:
        """
        评分单个景点对单个隐式需求的命中情况 (v3 - 语义相似度版本)
        返回 None 表示跳过检查
        返回 0-100 的分数
        """
        kw_lower = implicit_keyword.lower()
        keywords = ATTRACTION_FACILITY_KEYWORDS.get(kw_lower, [])
        special = SPECIAL_RULES.get(kw_lower, {}).get("attraction", {})
        
        facilities = attraction_info.get("facilities", [])
        restaurant_rating = attraction_info.get("restaurant_rating", 0)

        # foodie was scored on TWO dimensions — a food facility, and restaurant rating >= 3.5. The
        # rating half is dead: `rate_of_restaurant` is a deterministic function of `type` with only
        # four values in the whole KB (Nature 3.7, Historic 3.5, Museum 4.0, Theme Park 3.75), so
        # EVERY attraction clears 3.5 and the half was an unconditional +50. That is the same
        # "a cell ~100% of resources satisfy adds noise, not signal" rule already applied in
        # NON_DISCRIMINATIVE_CELLS, so the persona is judged on the facility alone.
        if special.get("dual_dimension"):
            return 100.0 if self._facility_hit(facilities, keywords) else 0.0

        if not keywords:
            return None
        return 100.0 if self._facility_hit(facilities, keywords) else 0.0
    
    def score_car(self, car_info: dict, implicit_keyword: str) -> Optional[float]:
        """
        评分单个租车对单个隐式需求的命中情况 (v3 - 语义相似度版本)
        返回 None 表示跳过检查
        返回 0-100 的分数
        """
        kw_lower = implicit_keyword.lower()
        keywords = CAR_FACILITY_KEYWORDS.get(kw_lower, [])
        
        # 无要求的情况：跳过检查
        if not keywords:
            return None
        
        extra_services = car_info.get("extra_services", [])
        return 100.0 if self._facility_hit(extra_services, keywords) else 0.0
    
    # Cells the KB cannot discriminate on, measured over the whole v2 KB. Scoring a cell that
    # ~100% (or ~0%) of resources satisfy adds noise, not signal: it awards the same credit to
    # every plan regardless of how well it matched the persona. 'fast-paced budget travel' x
    # attraction is satisfied by 100.0% of attractions ("city passes" / "luggage storage" /
    # "self-guided tours" are near-universal), so it is excluded and the persona is judged on
    # hotels instead (26.6% satisfaction — genuinely selective). Every other persona x type cell
    # sits in a healthy 5-63% band and is kept.
    NON_DISCRIMINATIVE_CELLS = {("fast-paced budget travel", "attraction")}

    def _persona_applies(self, persona: str, rtype: str) -> bool:
        """Does this persona place ANY requirement on this resource type the KB can discriminate?"""
        p = persona.lower()
        if (p, rtype) in self.NON_DISCRIMINATIVE_CELLS:
            return False
        if rtype == "hotel":
            return bool(HOTEL_FACILITY_KEYWORDS.get(p)) or bool(SPECIAL_RULES.get(p, {}).get("hotel"))
        if rtype == "attraction":
            return bool(ATTRACTION_FACILITY_KEYWORDS.get(p)) or bool(SPECIAL_RULES.get(p, {}).get("attraction"))
        if rtype == "car":
            return bool(CAR_FACILITY_KEYWORDS.get(p))
        return False

    def _resolve(self, item, kind, cities):
        if kind == "hotel":
            nm = item.get("name", "")
            for c in ([item.get("city")] if item.get("city") else []) + list(cities):
                info = self.db.get_hotel_info(nm, c)
                if info:
                    return info
            return None
        if kind == "attraction":
            nm = item.get("name", "")
            for c in ([item.get("city")] if item.get("city") else []) + list(cities):
                info = self.db.get_attraction_info(nm, c)
                if info:
                    return info
            return None
        ct = item.get("type") or item.get("car_type", "")
        claimed = item.get("price_per_day")
        for c in ([item.get("city")] if item.get("city") else []) + list(cities):
            info = self.db.get_car_info(ct, c, claimed_price=claimed)
            if info:
                return info
        return None

    def score_plan(self, plan_data: dict, implicit_keywords: list,
                   required_types: set = None, stay_cities: list = None) -> dict:
        """D1 = COVERAGE of persona needs, deterministic (no embeddings).

        For each persona and each resource TYPE the plan books that the persona cares about, the
        cell scores the BEST (max) booked resource of that type: 100 if it carries a persona-relevant
        facility (dual-dimension rules for luxury/foodie), 0 otherwise. A booked entity that does not
        resolve in the KB (a hallucination) scores 0 — it can never raise the cell. D1 is the mean
        over cells, /100. The max stops extra non-matching bookings from diluting the score (which
        unblocks itinerary density) and makes fabrication cost rather than pay.

        PER CITY. The max is taken WITHIN a stay city, then averaged ACROSS them — not once over the
        whole trip. A persona is a property of the traveller, so it has to hold everywhere they
        sleep: measured on a 3-city elderly-travellers itinerary, replacing five of six hotels with
        non-matching ones left D1 at a perfect 1.000, because one qualifying hotel anywhere on the
        trip carried every city. The generator only emits a task when a persona-satisfying option
        exists in EVERY stay city, so per-city satisfaction is exactly what the task was built to
        demand, and gold — which picks per city — is unaffected.
        """
        if not implicit_keywords:
            return {"total_score": None, "details": {}}

        cities = plan_data.get("cities", [])
        types = {
            "hotel": (plan_data.get("hotels", []), self.score_hotel),
            "attraction": (plan_data.get("attractions", []), self.score_attraction),
            "car": (plan_data.get("cars", []), self.score_car),
        }
        resolved = {t: [self._resolve(it, t, cities) for it in items] for t, (items, _fn) in types.items()}

        cells = []
        details = {"hotel_scores": [], "attraction_scores": [], "car_scores": []}
        required_types = required_types or set()
        for kw in implicit_keywords:
            for t, (items, fn) in types.items():
                if not self._persona_applies(kw, t):
                    continue                      # this persona has no stake in this resource type
                # Bucket this type's bookings by city, keeping the best in each.
                by_city = {}
                for it, info in zip(items, resolved[t]):
                    sc = fn(info, kw) if info is not None else 0.0
                    if sc is None:
                        sc = 0.0
                    ckey = _fold_city(it.get("city") or it.get("city_name") or "")
                    by_city[ckey] = max(by_city.get(ckey, 0.0), sc)
                    details[t + "_scores"].append({
                        t: (it.get("name") or it.get("type") or it.get("car_type", "")),
                        "keyword": kw, "score": sc, "resolved": info is not None,
                    })

                # APPLICABILITY COMES FROM THE TASK. For a type the task requires, every stay city
                # is a cell — one the plan books nothing in scores 0, because omission is an unmet
                # persona need, not an exemption (skipping it meant DELETING a non-matching hotel
                # raised D1 from 0.5 to 1.0). For a type the task did not require, only the cities
                # the plan actually books in are scored.
                if t in required_types and stay_cities:
                    for c in stay_cities:
                        cells.append(by_city.get(_fold_city(c), 0.0))
                elif by_city:
                    cells.extend(by_city.values())
                elif t in required_types:
                    cells.append(0.0)

        total = (sum(cells) / len(cells) / 100.0) if cells else None
        return {"total_score": total, "details": details, "num_checks": len(cells)}


# ============ 便捷函数 ============
_scorer: Optional[ImplicitRequirementsScorer] = None

def get_implicit_scorer() -> ImplicitRequirementsScorer:
    """获取评分器单例"""
    global _scorer
    if _scorer is None:
        _scorer = ImplicitRequirementsScorer()
    return _scorer


def score_d1_implicit_v2(plan_data: dict, implicit_keywords: list,
                         required_types: set = None, stay_cities: list = None) -> Optional[float]:
    """
    D1 隐式需求评分（v2 版本）
    
    Args:
        plan_data: 包含 hotels, attractions, cars, cities 的计划数据
        implicit_keywords: 隐式需求关键词列表
    
    Returns:
        0-1 的分数，None 表示无隐式需求
    """
    if not implicit_keywords:
        return None
    
    scorer = get_implicit_scorer()
    result = scorer.score_plan(plan_data, implicit_keywords, required_types, stay_cities)
    return result["total_score"]


# ============ 测试代码 ============
if __name__ == "__main__":
    # 测试评分逻辑
    db = ImplicitScoringDB()
    scorer = ImplicitRequirementsScorer(db)
    
    # 测试获取酒店信息
    hotel_info = db.get_hotel_info("Yan Garden Chaoyang", "Beijing")
    if hotel_info:
        print("=== 酒店信息 ===")
        print(f"Name: {hotel_info['name']}")
        print(f"Star: {hotel_info['star']}")
        print(f"Amenities: {hotel_info['amenities'][:5]}...")
        print(f"Restaurant Rating: {hotel_info['restaurant_rating']}")
        
        # 测试各种隐式需求评分
        print("\n=== 隐式需求评分 ===")
        test_keywords = ["luxury travelers", "with children", "foodie", "road trip"]
        for kw in test_keywords:
            score = scorer.score_hotel(hotel_info, kw)
            print(f"  {kw}: {score}")
    
    # 测试完整计划评分
    print("\n=== 计划评分测试 ===")
    mock_plan = {
        "hotels": [{"name": "Yan Garden Chaoyang"}],
        "attractions": [],
        "cars": [],
        "cities": ["Beijing"]
    }
    result = scorer.score_plan(mock_plan, ["luxury travelers", "road trip"])
    print(f"Total Score: {result['total_score']}")
    print(f"Num Checks: {result['num_checks']}")
    print(f"Details: {json.dumps(result['details'], indent=2)}")
