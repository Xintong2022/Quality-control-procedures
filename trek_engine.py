import copy
import json
import os
import sys

import requests

# ============================================================
# 引擎 · TREK 确定性评测接入
# TREK 是纯规则、离线、可复现的旅行规划评测器，评测"结构化行程计划"
# 对 TREK 自带合成知识库(375 城 / 航班·酒店·景点·租车)的匹配度。
# 仅依赖 pandas + numpy（本环境已具备）。
# ============================================================
TREK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trek_repo")
TREK_QUERIES_CSV = os.path.join(TREK_DIR, "trek_queries.csv")
TREK_GOLD_JSONL = os.path.join(TREK_DIR, "trek_gold.jsonl")

TREK_AVAILABLE = False
if os.path.isdir(TREK_DIR):
    if TREK_DIR not in sys.path:
        sys.path.insert(0, TREK_DIR)
    try:
        from data_loader import load_queries, get_sandbox_db
        from scoring import TravelPlanScorer, compute_aggregate_scores
        TREK_AVAILABLE = True
    except Exception as _e:  # 路径或依赖异常时，引擎 不可用但不影响引擎 B
        TREK_AVAILABLE = False
        print(f"[trek_engine] 引擎(TREK) 加载失败：{_e}")

# 维度中文名 + 类别中文名，仅用于展示
TREK_DIM_LABELS = [
    ("d0_keyword", "D0-key 请求实体满足"),
    ("d0_source", "D0-src 实体真实性(防幻觉)"),
    ("d1_implicit", "D1 隐式需求"),
    ("d2_unord_single", "D2 单城约束"),
    ("d2_unord_multi", "D2 多城约束"),
    ("d3_budget", "D3 预算有效"),
    ("d4_impossible", "D4 不可行拒绝"),
    ("d5_retry", "D5 工具效率"),
    ("b2_opening_hours", "B2 营业时间"),
    ("b3_spatiotemporal", "B3 时空可达"),
]
TREK_CAT_LABELS = [
    ("cat_satisfaction", "约束满足"),
    ("cat_truthfulness", "数据真实性"),
    ("cat_reasoning", "规划合理性"),
    ("cat_infeasibility", "不可行识别"),
    ("cat_efficiency", "效率"),
]


# 进程内单例缓存：Streamlit 单进程复用，且可无头测试（不依赖 streamlit runtime）
_SCORER = None
_QUERIES = None
_GOLD = None


def get_trek_scorer():
    global _SCORER
    if _SCORER is None:
        db = get_sandbox_db()
        _SCORER = TravelPlanScorer(db, {"beta": 4.0, "alpha": 0.5, "k_threshold": 15})
    return _SCORER


def get_trek_queries():
    global _QUERIES
    if _QUERIES is None:
        _QUERIES = load_queries(TREK_QUERIES_CSV)
    return _QUERIES


def get_trek_gold():
    global _GOLD
    if _GOLD is None:
        recs = []
        with open(TREK_GOLD_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        _GOLD = recs  # 行序即 query_index
    return _GOLD


def _inject_hallucination(plan_output: dict) -> None:
    """把一个真实景点改名为虚构名，演示引擎 抓幻觉（D0-src 下降）。"""
    plan = plan_output.get("plan")
    if not isinstance(plan, dict):
        return
    for day_key in sorted(plan.keys(), key=str):
        day = plan[day_key]
        if not isinstance(day, dict):
            continue
        attrs = day.get("attractions")
        if isinstance(attrs, list) and attrs:
            if isinstance(attrs[0], dict):
                attrs[0] = dict(attrs[0])
                attrs[0]["name"] = "【虚构景点】不存在的梦幻之地"
            break


def run_trek_engine(mode: str, sample_size: int, full: bool = False,
                    inject_defect: bool = False, mock_flaw: str = "无（完美）",
                    bot_struct_api: str = None):
    """返回 (per_task_list, aggregate_dict)。per_task = (idx, meta, plan_output, ScoreResult|None)。

    mode:
      - "gold": 直接评测 TREK Gold（离线确定性基准，预期 1.0）
      - "mock": 内置 Mock Bot——返回 Gold 计划但按 mock_flaw 注入缺陷以演示引擎 抓错
      - "bot" : 真实 Bot——POST 你的结构化 API，对返回的计划做确定性打分（方案 1 核心）
    """
    scorer = get_trek_scorer()
    queries = get_trek_queries()
    gold = get_trek_gold()
    n = len(queries)
    indices = list(range(n)) if full else list(range(min(sample_size, n)))

    per_task = []
    for idx in indices:
        meta = queries[idx]
        if mode == "bot":
            plan_output = call_structured_bot(bot_struct_api, meta, idx)
        else:
            rec = gold[idx] if idx < len(gold) else None
            if rec is None:
                plan_output = {"is_feasible": False, "plan": {}, "refusal_reason": ""}
            else:
                plan_output = {
                    "is_feasible": rec.get("is_feasible", False),
                    "plan": copy.deepcopy(rec.get("plan", {})),
                    "refusal_reason": rec.get("refusal_reason", ""),
                }
                if mode == "gold" and inject_defect and idx < 3 and not getattr(meta, "impossible", False):
                    _inject_hallucination(plan_output)
                elif mode == "mock":
                    apply_mock_flaw(plan_output, mock_flaw, meta)
        try:
            sr = scorer.score_query(plan_output, meta, tool_call_count=0)
            sr.query_index = idx
        except Exception:
            sr = None
        evidence = None
        if sr is not None:
            try:
                evidence = collect_evidence(scorer, plan_output, meta)
            except Exception:
                evidence = None
        per_task.append((idx, meta, plan_output, sr, evidence))

    valid = [t[3] for t in per_task if t[3] is not None]
    agg = compute_aggregate_scores(valid) if valid else {}
    return per_task, agg


def collect_evidence(scorer, plan_output: dict, meta) -> dict:
    """钻取：对单个任务采集逐维度判定证据，供 UI 展开「约束 → 判定 → 证据」。

    通过调用 scorer 的各 score_* 方法（与 score_query 同源逻辑）拿到内部判定值，
    并逐实体对照知识库 (db.verify_*) 列出幻觉清单，让 TREK 黑盒变得透明可审。
    """
    plan_data = scorer.extract_plan_data(plan_output)
    ev = {}

    # D3 预算：上限 vs 实际合计
    try:
        if getattr(meta, "budget", 0) and meta.budget > 0:
            s, cost = scorer.score_d3_budget(plan_data, meta)
            ev["d3_budget"] = {
                "budget": round(float(meta.budget), 2),
                "total_cost": round(float(cost), 2),
                "delta": round(float(cost) - float(meta.budget), 2),
                "score": s,
                "pass": (s is not None and s >= 0.999),
            }
        else:
            ev["d3_budget"] = {"budget": 0, "total_cost": 0, "delta": 0, "score": None, "pass": None}
    except Exception:
        ev["d3_budget"] = {"error": True}

    # D0-src 实体真实性（零容忍）：逐实体对照知识库
    try:
        hal = []
        for f in plan_data.get("flights", []) or []:
            fn = f.get("flight_number")
            if fn and not scorer.db.verify_flight(fn, f.get("departure_city"), f.get("arrival_city")):
                hal.append({"type": "航班", "name": str(fn), "city": f.get("arrival_city")})
        for h in plan_data.get("hotels", []) or []:
            n, c = h.get("name"), h.get("city")
            if n and (not c or not scorer.db.verify_hotel(n, c)):
                hal.append({"type": "酒店", "name": str(n), "city": c})
        for a in plan_data.get("attractions", []) or []:
            n, c = a.get("name"), a.get("city")
            if n and (not c or not scorer.db.verify_attraction(n, c)):
                hal.append({"type": "景点", "name": str(n), "city": c})
        for c in plan_data.get("cars", []) or []:
            t, cc = c.get("car_type") or c.get("type"), c.get("city")
            if t and (not cc or not scorer.db.verify_car(t, cc)):
                hal.append({"type": "租车", "name": str(t), "city": cc})
        s = scorer.score_d0_source(plan_data, meta)
        ev["d0_source"] = {
            "score": s,
            "hallucinations": hal,
            "pass": (s == 1.0) if s is not None else None,
        }
    except Exception:
        ev["d0_source"] = {"error": True}

    # B3 时空可达
    try:
        s = scorer.score_b3_spatiotemporal(plan_data)
        ev["b3"] = {"score": s, "pass": (s is not None and s >= 0.999)}
    except Exception:
        ev["b3"] = {"error": True}

    # B2 营业时间 / 闭馆日
    try:
        s = scorer.score_b2_opening_hours(plan_data, meta)
        ev["b2_opening_hours"] = {"score": s, "pass": (s is not None and s >= 0.999)}
    except Exception:
        ev["b2_opening_hours"] = {"error": True}

    # D2 城市约束
    try:
        d = scorer.score_d2_city(plan_data, meta)
        ev["d2"] = {"detail": d, "pass": (all(v == 1.0 for v in d.values()) if d else None)}
    except Exception:
        ev["d2"] = {"error": True}

    # D4 不可行识别
    try:
        s = scorer.score_d4_impossible(plan_data, meta)
        ev["d4"] = {
            "impossible": bool(getattr(meta, "impossible", False)),
            "is_feasible": plan_data.get("is_feasible"),
            "score": s,
            "pass": (s == 1.0) if s is not None else None,
        }
    except Exception:
        ev["d4"] = {"error": True}

    # D1 隐式需求
    try:
        s = scorer.score_d1_implicit(plan_data, meta)
        ev["d1"] = {"implicit": getattr(meta, "implicit_keywords", None), "score": s}
    except Exception:
        ev["d1"] = {"error": True}

    return ev


def apply_mock_flaw(plan_output: dict, flaw: str, meta) -> None:
    """对内置 Mock Bot 的计划注入缺陷，演示引擎 如何确定性地抓错。"""
    plan = plan_output.get("plan", {})
    if not isinstance(plan, dict):
        return
    if flaw == "可行任务却错误拒绝（全维度 0）":
        if not getattr(meta, "impossible", False):
            plan_output["is_feasible"] = False
            plan_output["refusal_reason"] = "（Mock 注入）抱歉，我无法为您规划此行程"
            plan_output["plan"] = {}
        return
    if flaw == "无（完美）":
        return
    # 在第一个含景点的 day 上动手
    for day_key in sorted(plan.keys(), key=str):
        day = plan.get(day_key)
        if not isinstance(day, dict):
            continue
        attrs = day.get("attractions") or day.get("attraction")
        if isinstance(attrs, list) and attrs:
            if flaw == "遗漏景点（D0-key↓）":
                attrs.pop()  # 删最后一个景点 → D0-key 实体满足率下降
                return
            if flaw == "虚构景点（D0-src↓）" and isinstance(attrs[0], dict):
                attrs[0] = dict(attrs[0])
                attrs[0]["name"] = "【虚构】不存在的梦幻之地XYZ"
                return


def _as_plan_output(data) -> dict:
    """容错解析 Bot 返回的 JSON：接受裸 payload 或 {plan_output|result|data|output:{...}} 包裹。"""
    if not isinstance(data, dict):
        return {"is_feasible": False, "plan": {}, "refusal_reason": "Bot 返回非 JSON 对象"}
    if "plan" in data or "is_feasible" in data or "refusal_reason" in data:
        return {
            "is_feasible": bool(data.get("is_feasible", False)),
            "refusal_reason": data.get("refusal_reason", ""),
            "plan": data.get("plan", {}) if isinstance(data.get("plan"), dict) else {},
        }
    for key in ("plan_output", "result", "data", "output"):
        if key in data and isinstance(data[key], dict):
            return _as_plan_output(data[key])
    return {"is_feasible": False, "plan": {}, "refusal_reason": "Bot 返回缺少 plan 字段"}


def call_structured_bot(url: str, meta, idx: int) -> dict:
    """方案 1 核心：把 TREK 查询发给你的结构化 Bot，取回 TREK 计划 JSON 供引擎 打分。"""
    payload = {
        "query": getattr(meta, "query", ""),
        "query_index": idx,
        "budget": getattr(meta, "budget", 0.0),
        "days": getattr(meta, "days", 0),
        "cities_count": getattr(meta, "cities_count", 1),
        "impossible": bool(getattr(meta, "impossible", False)),
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return _as_plan_output(r.json())
    except Exception as e:
        return {"is_feasible": False, "plan": {}, "refusal_reason": f"Bot 调用失败: {e}"}
