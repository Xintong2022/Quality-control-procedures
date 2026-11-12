# -*- coding: utf-8 -*-
"""
旅游 AI 质检平台 · 后端 API
--------------------------------
以用户提供的深色主题 HTML 为前端，本服务提供真实数据：
  - 案例库（cases.json）增删改查 + 上传（对齐《需求测试题库.xlsx》9 列）
  - 案例生成器（AI 造题）：从题库的服务类型/坑类型/目的地词表 + 真实示例，让 LLM 合成全新案例并入库
  - 测试构造器：把任意一条案例（含 AI 生成的）变成可执行测试（埋的坑→对抗输入，正确应对→期望清单）
  - 引擎：TREK 确定性评测（复用 trek_engine，离线、可复现、带逐维证据）—— 只评硬逻辑
  - 轻量裁判：LLM 盲审，只补 TREK 打不了的软质量（规避埋的坑 + 主动服务），不再与 TREK 并列
  - 看板聚合 / 执行记录 / 设置
端口 8501，静态页 index.html 由本服务托管。
"""
import os
import re
import json
import copy
import uuid
import threading
import datetime
import secrets
import hashlib
import hmac

from flask import Flask, request, jsonify, send_from_directory, Response, session

from trek_engine import (
    TREK_AVAILABLE, TREK_DIM_LABELS, TREK_CAT_LABELS,
    run_trek_engine, get_trek_scorer, get_trek_queries, get_trek_gold,
)
from pitfall_engine import run_pitfall_eval, FAMILIES as PITFALL_FAMILIES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_FILE = os.path.join(BASE_DIR, "cases.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
RUNS_FILE = os.path.join(BASE_DIR, "runs.json")
INDEX_FILE = os.path.join(BASE_DIR, "index.html")
PITFALL_VOCAB_FILE = os.path.join(BASE_DIR, "pitfall_vocab.json")

app = Flask(__name__, static_folder=None)

# 保护 cases/settings 读写
_lock = threading.Lock()

# ----------------------------------------------------------------------------
# 持久化辅助
# ----------------------------------------------------------------------------
def _read_json(path, default):
    if not os.path.exists(path):
        return copy.deepcopy(default)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return copy.deepcopy(default)

def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_cases():
    return _read_json(CASES_FILE, [])

def save_cases(cases):
    _write_json(CASES_FILE, cases)

def load_settings():
    s = _read_json(SETTINGS_FILE, {})
    s.setdefault("bot_api", "")
    s.setdefault("judge_model", "gpt-4o")
    s.setdefault("judge_base_url", "https://api.openai.com/v1")
    s.setdefault("judge_api_key", "")
    s.setdefault("judge_temperature", 0)
    s.setdefault("sample_times", 3)
    s.setdefault("generator_model", "")
    s.setdefault("admin_password_hash", "")
    return s

def save_settings(s):
    _write_json(SETTINGS_FILE, s)

def load_runs():
    return _read_json(RUNS_FILE, [])

def save_runs(runs):
    _write_json(RUNS_FILE, runs)


# ----------------------------------------------------------------------------
# 管理员登录 / 密码
# ----------------------------------------------------------------------------
def _load_or_create_secret():
    s = _read_json(SETTINGS_FILE, {})
    if not s.get("_secret_key"):
        s["_secret_key"] = secrets.token_hex(32)
        _write_json(SETTINGS_FILE, s)
    return s["_secret_key"]

app.secret_key = _load_or_create_secret()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

def _hash_password(pw):
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt), 100000)
    return "pbkdf2_sha256$" + salt + "$" + dk.hex()

def _verify_password(pw, stored):
    try:
        algo, salt, h = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), bytes.fromhex(salt), 100000)
        return hmac.compare_digest(dk.hex(), h)
    except Exception:
        return False

_AUTH_EXEMPT = {
    "/", "/api/health", "/api/auth/status", "/api/login", "/api/logout",
    "/api/admin/set-password", "/api/admin/remove-password",
}

@app.before_request
def _auth_guard():
    p = request.path
    if p in _AUTH_EXEMPT or not p.startswith("/api/"):
        return None
    s = load_settings()
    if not s.get("admin_password_hash"):
        return None  # 未设密码 → 开放模式（向后兼容）
    if session.get("admin_authed"):
        return None
    return jsonify({"ok": False, "error": "未登录或登录已失效", "need_login": True}), 401


# ----------------------------------------------------------------------------
# 案例模型解析（对齐《需求测试题库.xlsx》主表 9 列）
# ----------------------------------------------------------------------------
LEVEL1_KNOWN = [
    "机票", "签证", "酒店", "儿童", "自驾", "门票", "季节", "安全", "交通",
    "健康", "季节天气", "驾照", "费用", "高原", "宗教着装", "气候", "宗教",
    "行程", "着装",
]

def parse_pitfall_types(raw):
    """『门票+季节+机票+酒店』『交通(ZTL)+酒店』『酒店(年龄)+签证』→ 两级标签。"""
    if not raw or not str(raw).strip():
        return []
    out = []
    for tok in re.split(r"\s*\+\s*", str(raw).strip()):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", tok)
        if m:
            out.append({"level1": m.group(1).strip(), "level2": m.group(2).strip()})
        else:
            out.append({"level1": tok, "level2": ""})
    for it in out:
        l1 = it["level1"]
        hit = next((k for k in LEVEL1_KNOWN if k == l1 or k in l1 or l1 in k), None)
        if hit:
            it["level1"] = hit
    return out

def split_service_types(raw):
    if not raw or not str(raw).strip():
        return []
    if isinstance(raw, (list, tuple)):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).split("/") if s.strip()]

_TIME_PAT = re.compile(
    r"\d{4}\s*年|\d{1,2}\s*月|寒假|暑假|暑期|春节|元旦|国庆|五一|中秋|圣诞|"
    r"复活节|跨年|旺季|淡季|平日|周末|冬季|夏季|春季|秋季"
)

def derive_destination_period(scene):
    """从「场景」拆出 目的地 与 出行时段。

    场景 = 目的地×时段 的自由组合，例如「东京 11月」「意大利 罗马-佛罗伦萨」
    「欧洲 法意瑞」。优先按时间关键词切分；无时间关键词时整段作目的地、时段为空。
    """
    if not scene or not str(scene).strip():
        return "", ""
    s = str(scene).strip()
    matches = _TIME_PAT.findall(s)
    if matches:
        period = " ".join(matches)
        dest = _TIME_PAT.sub("", s)
        dest = re.sub(r"[\s·\-—_]+", " ", dest).strip()
        if not dest:
            dest = s
        return dest, period
    return s, ""

def build_case_from_bank_row(r):
    """把题库主表一行（dict，列名同 Excel）转成新案例模型。"""
    def g(name):
        return r.get(name)
    scene = str(g("场景") or "").strip()
    dest, period = derive_destination_period(scene)
    pit = parse_pitfall_types(g("坑类型"))
    return {
        "case_id": str(g("编号") or "").strip(),
        "scene": scene,
        "destination": dest,
        "travel_period": period,
        "user_prompt": str(g("客户原话(测试输入)") or g("客户原话") or "").strip(),
        "travel_time": str(g("出行时间") or "").strip(),
        "group_size": str(g("人数") or "").strip(),
        "service_types": split_service_types(g("服务类型")),
        "pitfall_types": pit,
        "buried_pitfalls": str(g("埋的坑") or g("埋的坑(评分用·不喂AI)") or "").strip(),
        "correct_response": str(g("正确应对") or g("正确应对(评分用·不喂AI)") or "").strip(),
        "category": dest or "未分类",
        "hard_assertions": {"max_budget": None, "max_daily_walking_km": None},
        "known_pitfalls": sorted({p["level1"] for p in pit}),
        "source": "需求测试题库",
        "gen_method": "上传导入",
        "zone": "A+B",
        "recent_score": None,
    }

# ----------------------------------------------------------------------------
# 引擎：把 TREK 证据映射成报告页「硬断言明细」的 5 类
# ----------------------------------------------------------------------------
def map_evidence_to_assertions(ev, meta, case):
    """把 collect_evidence() 返回的逐维证据，归并到报告页的 5 类硬断言。"""
    def fmt_pass(p):
        return "通过" if p else "未通过"

    budget = ev.get("d3_budget", {})
    src = ev.get("d0_source", {})
    b2 = ev.get("b2_opening_hours", {})
    b3 = ev.get("b3", {})
    d4 = ev.get("d4_impossible", {})

    # 案例级预算硬约束（优先用案例自己的 max_budget）
    case_budget = (case or {}).get("hard_assertions", {}).get("max_budget")
    limit = budget.get("limit") or case_budget
    actual = budget.get("actual")
    budget_pass = budget.get("pass")
    if limit is not None and actual is not None and budget_pass is None:
        budget_pass = actual <= limit

    walk_limit = (case or {}).get("hard_assertions", {}).get("max_daily_walking_km")

    hal = src.get("hallucinations", []) or []
    hallucinations_txt = "；".join(
        f"「{h.get('name','?')}」({h.get('city','?')}) 不在知识库" for h in hal
    ) if hal else "0 虚构"

    impossible = getattr(meta, "impossible", False) if meta is not None else False

    return [
        {
            "name": "预算求和与限额",
            "caliber": f"max_budget = {limit}（含大交通）" if limit is not None else "未配置预算上限",
            "measured": f"{actual}" if actual is not None else "—",
            "pass": bool(budget_pass) if budget_pass is not None else None,
            "detail": "预算总额在限额内" if (budget_pass) else ("超出限额" if budget_pass is False else "N/A"),
        },
        {
            "name": "POI 实体真值匹配",
            "caliber": "must_verify_poi_existence = true · 知识库对照",
            "measured": f"{src.get('valid', 0)} 命中 / {len(hal)} 虚构" if src else "—",
            "pass": (len(hal) == 0) if src else None,
            "detail": hallucinations_txt if hal else ("全部实体命中知识库" if src else "N/A"),
        },
        {
            "name": "营业时间与可达性",
            "caliber": "闭馆日 · 营业时间窗 · 门到门转场时间",
            "measured": f"b2={'通过' if b2.get('pass') else '未通过'} · b3={'通过' if b3.get('pass') else '未通过'}",
            "pass": (bool(b2.get("pass", False)) and bool(b3.get("pass", False))) if (b2 or b3) else None,
            "detail": (b3.get("detail") or b2.get("detail") or "时空可达性校验") if (b2 or b3) else "N/A",
        },
        {
            "name": "单日步行强度上限",
            "caliber": f"max_daily_walking_limit_km = {walk_limit}" if walk_limit is not None else "未配置步行上限",
            "measured": "见 TREK b3 时空可达" if walk_limit is not None else "—",
            "pass": None,
            "detail": "TREK 基准为合成域，无逐日步行里程；业务行程需在结构化计划里带距离字段后由案例级断言校验",
        },
        {
            "name": "拒绝能力测试",
            "caliber": f"expect_infeasibility_refusal = {impossible}",
            "measured": f"d4={'通过' if d4.get('pass') else '未通过'}" if d4 else "不可行任务" if impossible else "需求可行",
            "pass": bool(d4.get("pass")) if d4 else None,
            "detail": "需求可行时校验未误拒；不可行时校验明确拒绝" if d4 else ("不可行任务（应拒答）" if impossible else "N/A"),
        },
    ]


# ----------------------------------------------------------------------------
# 案例生成器：根据一条案例产出「可执行测试」（驱动 Bot 的输入 + 期望/规避清单）
# 这是用户心智模型里缺失的那一步：案例库 → 生成器造题 → 驱动 Bot → TREK 评硬 + 轻量裁判补软
# ----------------------------------------------------------------------------
def _split_sentences(text):
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[；;。\n\r]+", str(text))
    return [p.strip() for p in parts if p.strip()]


def build_test_from_case(case):
    """案例生成器：吃一条案例，产出可执行测试。

    字段接线（核心，按用户要求把字段接对）：
      - user_prompt           → 测试基础输入（Bot 可见，A 区）
      - buried_pitfalls       → 既造「对抗性客户输入」，又作轻量裁判「应规避」底牌（双重消费）
      - correct_response      → 轻量裁判「期望做到」清单（底牌）
      - service_types/pitfall_types/scene → 决定测试类型与造题侧重
      - travel_time/group_size       → 行程规划类造题的约束参数
    """
    base = (case.get("user_prompt") or "").strip()
    buried = (case.get("buried_pitfalls") or "").strip()
    correct = (case.get("correct_response") or "").strip()
    svc = case.get("service_types") or []
    if isinstance(svc, str):
        svc = [s.strip() for s in svc.split("/") if s.strip()]
    pits = case.get("pitfall_types") or []
    pit_labels = [(p.get("level1") if isinstance(p, dict) else str(p)) for p in pits]
    travel_time = (case.get("travel_time") or "").strip()
    group_size = (case.get("group_size") or "").strip()

    # 测试类型：有出行时间/人数/行程类服务 → 行程规划类；否则服务质检类
    itinerary_svc = any(k in " ".join(svc) for k in ["酒店", "机票", "门票", "包车", "自驾", "租车", "行程", "机酒"])
    is_itinerary = bool(travel_time or group_size or itinerary_svc)
    test_type = "itinerary" if is_itinerary else "service_qa"

    # 埋的坑 → 对抗性客户输入（把坑翻译成客户会怎么试探/表述）+ 裁判「应规避」清单
    avoid_list = _split_sentences(buried)
    probes = ["【模拟客户·按埋点施压】" + s for s in avoid_list]
    # 正确应对 → 期望做到的行为清单
    expected_behaviors = _split_sentences(correct)

    # 组装发给 Bot 的测试输入（在原始客户原话基础上注入对抗点，透明标记）
    generated_query = base
    if probes:
        generated_query = base + "\n\n[质检生成器注入·模拟客户追加需求]\n" + "\n".join(probes)

    constraints = {
        "travel_time": travel_time or None,
        "group_size": group_size or None,
        "service_types": svc,
        "scene": (case.get("scene") or "").strip() or None,
        "pitfall_types": pit_labels,
    }

    return {
        "test_type": test_type,
        "base_query": base,
        "generated_query": generated_query,
        "adversarial_probes": probes,
        "avoid_list": avoid_list,
        "expected_behaviors": expected_behaviors,
        "itinerary_constraints": constraints,
        "source_case_id": case.get("case_id"),
    }


# ----------------------------------------------------------------------------
# 案例生成器（AI 造题）：从题库词表 + 真实示例，让 LLM 合成全新案例，回写入案例库
# ----------------------------------------------------------------------------
import random as _rnd

def load_pitfall_vocab():
    """读取持久化的坑类型词表（由 pitfall_vocab.json 维护，可手动扩展）。
    返回 {by_level1, level1, level2}；文件不存在时返回 None。"""
    if os.path.exists(PITFALL_VOCAB_FILE):
        try:
            return json.load(open(PITFALL_VOCAB_FILE, encoding="utf-8"))
        except Exception:
            return None
    return None


def save_pitfall_vocab(by_level1):
    """把 {一级: [二级,...]} 规整并写回 pitfall_vocab.json，自动派生扁平的一级/二级列表。
    同一二级若挂在多个一级下，扁平列表只保留一份（去重）。"""
    clean = {}
    for k, vals in (by_level1 or {}).items():
        k = (str(k) or "").strip()
        if not k:
            continue
        items, seen = [], set()
        for v in (vals or []):
            v = (str(v) or "").strip()
            if not v or v in seen:
                continue
            seen.add(v)
            items.append(v)
        clean[k] = items
    flat2 = sorted({v for items in clean.values() for v in items})
    data = {"by_level1": clean, "level1": sorted(clean.keys()), "level2": flat2}
    with _lock:
        _write_json(PITFALL_VOCAB_FILE, data)
    return data


def _vocab_library_pairs(cases):
    """从案例库统计实际使用的 一级→[二级] 配对，以及每个一级的出现次数。

    注意：一级即使没有二级（如仅「机票」），也必须出现在配对表里（值为空列表），
    否则「按案例库重建」会把这些一级类型整体丢掉。"""
    pairs, counts = {}, {}
    for c in cases or []:
        for p in (c.get("pitfall_types") or []):
            if not isinstance(p, dict):
                continue
            l1 = (p.get("level1") or "").strip()
            l2 = (p.get("level2") or "").strip()
            if not l1:
                continue
            counts[l1] = counts.get(l1, 0) + 1
            bucket = pairs.setdefault(l1, [])
            if l2 and l2 not in bucket:
                bucket.append(l2)
    return pairs, counts


def _collect_library_vocab(cases, max_examples=4):
    """从现有题库提炼出题词表 + 少量真实示例，供 LLM 照着套路出新题。
    坑类型词表会与持久化的 pitfall_vocab.json 合并（以后者为准扩展覆盖面）。"""
    svc, l1, l2, dests, scenes = set(), set(), set(), set(), set()
    for c in cases:
        for s in (c.get("service_types") or []):
            if isinstance(s, str):
                svc.add(s.strip())
        for p in (c.get("pitfall_types") or []):
            if isinstance(p, dict):
                if p.get("level1"):
                    l1.add(p["level1"])
                if p.get("level2"):
                    l2.add(p["level2"])
            elif isinstance(p, str):
                l1.add(p.strip())
        if c.get("destination"):
            dests.add(c["destination"])
        if c.get("scene"):
            scenes.add(c["scene"])

    vocab_file = load_pitfall_vocab()
    by_level1 = {}
    if vocab_file:
        for k, v in (vocab_file.get("by_level1") or {}).items():
            l1.add(k)
            for lv in v:
                l2.add(lv)
            by_level1[k] = list(v)
        for x in (vocab_file.get("level1") or []):
            l1.add(x)
        for x in (vocab_file.get("level2") or []):
            l2.add(x)

    ex = _rnd.sample(cases, min(max_examples, len(cases))) if cases else []
    return {
        "service_types": sorted(svc),
        "pitfall_level1": sorted(l1),
        "pitfall_level2": sorted(l2),
        "pitfall_by_level1": by_level1,
        "destinations": sorted(dests)[:80],
        "scenes": sorted(scenes)[:50],
        "examples": ex,
    }


def _normalize_generated_case(rc, cid):
    """把模型返回的一条原始 dict 规整成与 add_case 一致的案例 schema。"""
    if not isinstance(rc, dict):
        return None
    svc_raw = rc.get("service_types") or []
    if isinstance(svc_raw, str):
        svc = [s.strip() for s in re.split(r"[/、，,；;]", svc_raw) if s.strip()]
    else:
        svc = [str(s).strip() for s in svc_raw if str(s).strip()]

    pits_raw = rc.get("pitfall_types") or []
    pits = []
    for p in pits_raw:
        if isinstance(p, dict):
            l1 = (p.get("level1") or "").strip()
            l2 = (p.get("level2") or "").strip()
            if l1 or l2:
                pits.append({"level1": l1, "level2": l2})
        elif isinstance(p, str) and p.strip():
            pits.append({"level1": p.strip(), "level2": ""})

    scene = (rc.get("scene") or "").strip()
    dest = (rc.get("destination") or "").strip() or derive_destination_period(scene)[0]
    period = (rc.get("travel_period") or "").strip() or derive_destination_period(scene)[1]
    up = (rc.get("user_prompt") or "").strip()
    if not up:
        return None
    return {
        "case_id": cid,
        "scene": scene,
        "destination": dest,
        "travel_period": period,
        "user_prompt": up,
        "travel_time": (rc.get("travel_time") or "").strip(),
        "group_size": (rc.get("group_size") or "").strip(),
        "service_types": svc,
        "pitfall_types": pits,
        "buried_pitfalls": (rc.get("buried_pitfalls") or "").strip(),
        "correct_response": (rc.get("correct_response") or "").strip(),
        "category": rc.get("category") or dest or "未分类",
        "hard_assertions": {"max_budget": None, "max_daily_walking_km": None},
        "known_pitfalls": sorted({p.get("level1") for p in pits if p.get("level1")}),
        "source": "AI生成",
        "gen_method": "LLM生成",
        "zone": "A+B",
        "recent_score": None,
    }


def generate_new_cases(cfg, n=5, filters=None):
    """调用 LLM 合成 n 条全新案例，并持久化追加进 cases.json。"""
    from openai import OpenAI
    key = (cfg or {}).get("judge_api_key") or ""
    base = (cfg or {}).get("judge_base_url") or "https://api.openai.com/v1"
    model = (cfg or {}).get("generator_model") or (cfg or {}).get("judge_model") or "gpt-4o"
    if not key:
        return {"ok": False, "error": "未配置生成模型 Key：请在「设置」填写裁判/生成 Key 后再造题。", "cases": []}

    cases = load_cases()
    vocab = _collect_library_vocab(cases, max_examples=4)

    examples_json = json.dumps(
        [{
            "scene": e.get("scene"), "destination": e.get("destination"),
            "travel_period": e.get("travel_period"), "user_prompt": e.get("user_prompt"),
            "travel_time": e.get("travel_time"), "group_size": e.get("group_size"),
            "service_types": e.get("service_types"), "pitfall_types": e.get("pitfall_types"),
            "buried_pitfalls": e.get("buried_pitfalls"), "correct_response": e.get("correct_response"),
        } for e in vocab["examples"]], ensure_ascii=False, indent=2)

    sys_prompt = (
        "你是海外旅游顾问质检平台的「案例生成器」。你的任务是从已有测试题库中提炼出题套路，"
        "合成全新的、彼此不重复的测试案例，用于考核旅游顾问 AI（Bot）。\n"
        "每个案例必须包含：\n"
        "  - scene: 场景名（如「东京亲子温泉」「清迈旅拍」）\n"
        "  - destination: 目的地（城市/国家）\n"
        "  - travel_period: 出行时段（如「7-8月暑期」「元旦」）\n"
        "  - user_prompt: 模拟客户原话（自然中文，含真实需求与隐含诉求，可带情绪/模糊表述）\n"
        "  - travel_time: 出行时间/天数（行程类案例必填，服务类可空）\n"
        "  - group_size: 同行人数（如「2大1小」）\n"
        "  - service_types: 涉及的服务类型数组，从给定词表选取（如 [\"民宿\",\"地陪\",\"票务\"]）\n"
        "  - pitfall_types: 坑类型数组，每项 {level1:一级, level2:二级}，从给定词表选取\n"
        "  - buried_pitfalls: 该案例埋的坑（客户易踩、Bot 易翻车点，换行分隔多条）\n"
        "  - correct_response: 优秀顾问的正确应对（换行分隔多条，对应规避埋的坑）\n"
        "要求：\n"
        "  1) 严格输出 JSON：{\"cases\":[...]}，不要任何额外说明。\n"
        "  2) 每条案例都是全新场景，禁止照抄示例原文。\n"
        "  3) 跨给定词表覆盖不同 目的地/服务/坑类型 组合，提升多样性。\n"
        "  4) 行程规划类案例须给 travel_time 与 group_size；纯服务咨询类可空。\n"
        "  5) buried_pitfalls 与 correct_response 要具体、可质检，而非泛泛而谈。"
    )
    by_level1_str = json.dumps(vocab.get("pitfall_by_level1") or {}, ensure_ascii=False)
    user_prompt = (
        f"题库词表（生成时请在 service_types / pitfall_types 中优先使用这些取值）：\n"
        f"服务类型：{json.dumps(vocab['service_types'], ensure_ascii=False)}\n"
        f"坑类型一级：{json.dumps(vocab['pitfall_level1'], ensure_ascii=False)}\n"
        f"坑类型二级：{json.dumps(vocab['pitfall_level2'], ensure_ascii=False)}\n"
        f"坑类型一级→二级 配对参考（pitfall_types 的 level1/level2 请尽量按此对应，避免乱配）：\n{by_level1_str}\n"
        f"目的地候选：{json.dumps(vocab['destinations'], ensure_ascii=False)}\n"
        f"场景候选：{json.dumps(vocab['scenes'], ensure_ascii=False)}\n\n"
        f"以下是 4 条真实示例（仅供学习格式与风格，禁止复制其内容）：\n{examples_json}\n\n"
        f"请生成 {n} 条全新测试案例（严格 JSON，键名与示例一致）。"
    )
    if filters:
        extra = []
        if filters.get("service"):
            extra.append(f"重点覆盖服务类型：{filters['service']}")
        if filters.get("pitfall"):
            extra.append(f"重点覆盖坑类型一级：{filters['pitfall']}")
        if extra:
            user_prompt += "\n补充侧重：" + "；".join(extra)

    try:
        client = OpenAI(api_key=key, base_url=base)
        resp = client.chat.completions.create(
            model=model, temperature=0.9,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        obj = json.loads(raw)
        arr = obj.get("cases") if isinstance(obj, dict) and isinstance(obj.get("cases"), list) else (obj if isinstance(obj, list) else [])
    except Exception as e:
        return {"ok": False, "error": f"生成调用失败：{e}", "cases": []}

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    generated = []
    for i, rc in enumerate(arr[:n]):
        c = _normalize_generated_case(rc, f"GEN-{now}-{i+1}")
        if c:
            generated.append(c)

    if not generated:
        return {"ok": False, "error": "模型未返回有效案例（可能格式不符），请重试。", "cases": []}

    with _lock:
        allc = load_cases()
        allc.extend(generated)
        save_cases(allc)

    return {"ok": True, "count": len(generated), "cases": generated}


@app.route("/api/generate", methods=["POST"])
def generate_test():
    """把一条案例变成可执行测试（供前端预览 / 报告流直接消费）。"""
    data = request.get_json(force=True, silent=True) or {}
    case_id = data.get("case_id", "")
    cases = load_cases()
    case = next((c for c in cases if c.get("case_id") == case_id), None)
    if not case:
        return jsonify({"ok": False, "error": f"未找到案例 {case_id}"}), 404
    try:
        test = build_test_from_case(case)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "test": test})


@app.route("/api/generate/cases", methods=["POST"])
def generate_cases_api():
    """AI 造题：从题库词表合成 n 条全新案例并入库。"""
    data = request.get_json(force=True, silent=True) or {}
    try:
        n = int(data.get("n", 5))
    except Exception:
        n = 5
    n = max(1, min(n, 50))
    filters = data.get("filters")
    cfg = load_settings()
    res = generate_new_cases(cfg, n, filters)
    return jsonify(res), (200 if res.get("ok") else 400)


# ----------------------------------------------------------------------------
# 轻量裁判（补 TREK 盲区）：LLM 盲审，只评软性服务质量（OpenAI 兼容）
# ----------------------------------------------------------------------------
def run_engine_b(case, cfg, test=None):
    """轻量裁判（补 TREK 盲区）：只评软性服务质量——规避埋的坑 + 做到正确应对。

    不再作为与 TREK 并列的『第二评分器』，而是对 TREK 打不了的那部分做盲审。
    测试输入来自案例生成器 build_test_from_case（用埋的坑造对抗输入）。
    """
    from openai import OpenAI

    bot_api = (cfg or {}).get("bot_api") or ""
    judge_key = (cfg or {}).get("judge_api_key") or ""
    judge_model = (cfg or {}).get("judge_model") or "gpt-4o"
    judge_base = (cfg or {}).get("judge_base_url") or "https://api.openai.com/v1"

    if test is None:
        test = build_test_from_case(case)
    query = test.get("generated_query") or case.get("user_prompt", "")

    result = {
        "enabled": True,
        "role": "lightweight_judge",
        "scope": "soft_quality_only",
        "bot_text": "",
        "scores": None,
        "pitfalls": [],
        "missed_expected": [],
        "summary": "",
        "fatal": False,
        "error": None,
    }

    # 1) 调 Bot 拿文本回答（用的是生成器产出的测试输入，含对抗注入）
    bot_text = ""
    if bot_api:
        try:
            import requests
            r = requests.post(bot_api, json={
                "query": query,
                "case_id": case.get("case_id"),
                "test_type": test.get("test_type"),
            }, timeout=30)
            r.raise_for_status()
            data = r.json()
            bot_text = data.get("bot_text") or data.get("answer") or data.get("text") or data.get("bot_output_solution") or ""
            if not isinstance(bot_text, str):
                bot_text = json.dumps(bot_text, ensure_ascii=False)
        except Exception as e:
            result["error"] = f"Bot 调用失败：{e}"
    else:
        bot_text = "（未配置 Bot API，以下为模拟回答占位）" + query

    result["bot_text"] = bot_text

    # 2) 轻量裁判：只评软质量（避坑 + 主动服务），不重复评硬性行程排期/预算
    if not judge_key:
        result["error"] = (result["error"] + "；" if result["error"] else "") + "未配置裁判模型 Key，轻量裁判仅展示 Bot 回答"
        return result

    try:
        client = OpenAI(api_key=judge_key, base_url=judge_base)
        # 底牌直接取生成器切好的清单（埋的坑=应规避，正确应对=期望做到），不再重复解析原始字段
        avoid = test.get("avoid_list") or []
        expected = test.get("expected_behaviors") or []
        avoid_txt = "\n".join(f"{i+1}. {x}" for i, x in enumerate(avoid)) if avoid else "（无）"
        expected_txt = "\n".join(f"{i+1}. {x}" for i, x in enumerate(expected)) if expected else "（无）"
        sys_prompt = (
            "你是旅游质检的「轻量裁判」，只负责评估 TREK 确定性评测覆盖不到的那部分——软性服务质量；"
            "不要重复评硬性行程排期/预算/可行性（那是 TREK 的职责）。\n"
            "请评估两点：\n"
            "1) 避坑风控：Bot 是否识别并妥善规避/提示【埋的坑(应规避)】清单里的风险；\n"
            "2) 主动服务：Bot 是否做到了【正确应对(期望做到)】清单里的主动服务动作。\n"
            "严格只输出 JSON："
            '{"pitfall_avoidance":<0-10,可一位小数>,"proactive_service":<0-10>,'
            '"has_fatal_risk":<bool>,'
            '"detected_pitfalls":[{"pitfall_category":<str>,"severity":<"严重"|"轻微">,"matched_avoid":<bool>,"suggestion":<str>}],'
            '"missed_expected":[<str>...],"summary":<str>}'
        )
        user_prompt = (
            f"原始用户需求：{case.get('user_prompt','')}\n"
            f"发给 Bot 的测试输入（含质检生成器注入的对抗点）：{query}\n"
            f"Bot 回答：{bot_text}\n"
            f"【埋的坑·应规避·底牌】\n{avoid_txt}\n"
            f"【正确应对·期望做到·底牌】\n{expected_txt}\n"
            f"硬约束（仅参考，不纳入你的评分）：{json.dumps(case.get('hard_assertions',{}), ensure_ascii=False)}"
        )
        resp = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        pa = float(parsed.get("pitfall_avoidance", 0))
        ps = float(parsed.get("proactive_service", 0))
        overall = round(pa * 0.5 + ps * 0.5, 1)
        result["scores"] = {
            "pitfall_avoidance": pa,
            "proactive_service": ps,
            "overall": overall,
        }
        result["fatal"] = bool(parsed.get("has_fatal_risk", False))
        result["pitfalls"] = parsed.get("detected_pitfalls", [])
        result["missed_expected"] = parsed.get("missed_expected", [])
        result["summary"] = parsed.get("summary", "")
    except Exception as e:
        result["error"] = (result["error"] + "；" if result["error"] else "") + f"轻量裁判调用失败：{e}"
    return result


# ----------------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    if os.path.exists(INDEX_FILE):
        return send_from_directory(BASE_DIR, "index.html")
    return "index.html not found. Please place the frontend at travel_qa_system/index.html", 404


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "trek_available": TREK_AVAILABLE,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    })


# ---------------- 案例库 ----------------
@app.route("/api/cases", methods=["GET"])
def get_cases():
    cases = load_cases()
    return jsonify({"ok": True, "count": len(cases), "cases": cases})


@app.route("/api/cases", methods=["POST"])
def add_case():
    data = request.get_json(force=True, silent=True) or {}
    case_id = (data.get("case_id") or "").strip()
    if not case_id:
        return jsonify({"ok": False, "error": "case_id 不能为空"}), 400
    cases = load_cases()
    if any(c.get("case_id") == case_id for c in cases):
        return jsonify({"ok": False, "error": f"case_id={case_id} 已存在"}), 409
    scene = (data.get("scene") or "").strip()
    dest = (data.get("destination") or "").strip() or derive_destination_period(scene)[0]
    period = (data.get("travel_period") or "").strip() or derive_destination_period(scene)[1]
    pit = data.get("pitfall_types") or parse_pitfall_types(data.get("pitfall_types_raw", ""))
    ha = data.get("hard_assertions") or {}
    if isinstance(ha, str):
        try:
            ha = json.loads(ha)
        except Exception:
            ha = {}
    case = {
        "case_id": case_id,
        "scene": scene,
        "destination": dest,
        "travel_period": period,
        "user_prompt": data.get("user_prompt", ""),
        "travel_time": data.get("travel_time", ""),
        "group_size": data.get("group_size", ""),
        "service_types": split_service_types(data.get("service_types", [])),
        "pitfall_types": pit,
        "buried_pitfalls": data.get("buried_pitfalls", ""),
        "correct_response": data.get("correct_response", ""),
        "category": data.get("category") or dest or "未分类",
        "hard_assertions": ha or {"max_budget": None, "max_daily_walking_km": None},
        "known_pitfalls": data.get("known_pitfalls") or sorted({p.get("level1") for p in pit if p.get("level1")}),
        "source": data.get("source", "business"),
        "gen_method": data.get("gen_method", "手动新增"),
        "zone": data.get("zone", "A+B"),
        "recent_score": data.get("recent_score", None),
    }
    with _lock:
        cases.append(case)
        save_cases(cases)
    return jsonify({"ok": True, "case": case})


@app.route("/api/cases/upload", methods=["POST"])
def upload_cases():
    # 支持 multipart 文件（.json/.csv/.xlsx）或 raw JSON body
    cases = load_cases()
    added, skipped = [], []
    try:
        if "file" in request.files:
            import io, csv
            f = request.files["file"]
            fname = (f.filename or "").lower()
            raw = f.read()
            if fname.endswith(".xlsx"):
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
                ws = wb[wb.sheetnames[0]]
                rows = list(ws.iter_rows(values_only=True))
                header = [str(h).strip() if h is not None else "" for h in rows[0]]
                rows = [dict(zip(header, r)) for r in rows[1:] if r is not None]
                for r in rows:
                    cid = str(r.get("编号") or "").strip()
                    if not cid:
                        skipped.append({"reason": "缺编号"})
                        continue
                    if any(c.get("case_id") == cid for c in cases):
                        skipped.append({"reason": "已存在", "case_id": cid})
                        continue
                    case = build_case_from_bank_row(r)
                    if not case["case_id"]:
                        continue
                    cases.append(case)
                    added.append(cid)
            elif fname.endswith(".csv"):
                rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="ignore"))))
                for r in rows:
                    cid = (r.get("case_id") or "").strip()
                    if not cid:
                        skipped.append({"reason": "缺 case_id"})
                        continue
                    if any(c.get("case_id") == cid for c in cases):
                        skipped.append({"reason": "已存在", "case_id": cid})
                        continue
                    cases.append(_normalize_case(r))
                    added.append(cid)
            else:  # json
                rows = json.loads(raw.decode("utf-8-sig", errors="ignore"))
                for r in rows:
                    cid = (r.get("case_id") or "").strip()
                    if not cid:
                        skipped.append({"reason": "缺 case_id"})
                        continue
                    if any(c.get("case_id") == cid for c in cases):
                        skipped.append({"reason": "已存在", "case_id": cid})
                        continue
                    cases.append(_normalize_case(r))
                    added.append(cid)
        else:
            rows = request.get_json(force=True, silent=True) or []
            for r in rows:
                cid = (r.get("case_id") or "").strip()
                if not cid:
                    skipped.append({"reason": "缺 case_id"})
                    continue
                if any(c.get("case_id") == cid for c in cases):
                    skipped.append({"reason": "已存在", "case_id": cid})
                    continue
                cases.append(_normalize_case(r))
                added.append(cid)
        with _lock:
            save_cases(cases)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "added": added, "skipped": skipped, "total": len(cases)})


def _normalize_case(r):
    """把 JSON / CSV 上传的案例行归一到新模型（兼容新旧字段名）。"""
    scene = (r.get("scene") or "").strip()
    dest, period = derive_destination_period(scene)
    dest = (r.get("destination") or "").strip() or dest
    period = (r.get("travel_period") or "").strip() or period
    pit = r.get("pitfall_types") or parse_pitfall_types(r.get("pitfall_types_raw") or r.get("pitfall_types") or "")
    ha = r.get("hard_assertions") or {}
    if isinstance(ha, str):
        try:
            ha = json.loads(ha)
        except Exception:
            ha = {}
    kp = r.get("known_pitfalls") or []
    if isinstance(kp, str):
        kp = [x.strip() for x in kp.split("|") if x.strip()]
    return {
        "case_id": (r.get("case_id") or "").strip(),
        "scene": scene,
        "destination": dest,
        "travel_period": period,
        "user_prompt": r.get("user_prompt", ""),
        "travel_time": r.get("travel_time", ""),
        "group_size": r.get("group_size", ""),
        "service_types": split_service_types(r.get("service_types", [])),
        "pitfall_types": pit,
        "buried_pitfalls": r.get("buried_pitfalls", ""),
        "correct_response": r.get("correct_response", ""),
        "category": r.get("category") or dest or "未分类",
        "hard_assertions": ha or {"max_budget": None, "max_daily_walking_km": None},
        "known_pitfalls": kp or sorted({p.get("level1") for p in pit if p.get("level1")}),
        "source": r.get("source", "business"),
        "gen_method": r.get("gen_method", "上传导入"),
        "zone": r.get("zone", "A+B"),
        "recent_score": r.get("recent_score", None),
    }


@app.route("/api/cases/<case_id>", methods=["DELETE"])
def delete_case(case_id):
    cases = load_cases()
    new = [c for c in cases if c.get("case_id") != case_id]
    if len(new) == len(cases):
        return jsonify({"ok": False, "error": "未找到该案例"}), 404
    with _lock:
        save_cases(new)
    return jsonify({"ok": True})


# ---------------- 坑类型词表（造题用，可预览 / 编辑） ----------------
@app.route("/api/vocab", methods=["GET"])
def vocab_get():
    """返回当前坑类型词表 + 案例库中的实际使用分布（用于标注哪些是题库里真出现过的）。"""
    vocab = load_pitfall_vocab() or {"by_level1": {}, "level1": [], "level2": []}
    cases = load_cases()
    lib_pairs, lib_counts = _vocab_library_pairs(cases)
    return jsonify({
        "ok": True,
        "by_level1": vocab.get("by_level1") or {},
        "level1": vocab.get("level1") or sorted(vocab.get("by_level1", {}).keys()),
        "level2": vocab.get("level2") or [],
        "library_pairs": lib_pairs,
        "library_counts": lib_counts,
        "stats": {
            "level1_count": len(vocab.get("by_level1") or {}),
            "level2_count": len(vocab.get("level2") or []),
            "pair_count": sum(len(v) for v in (vocab.get("by_level1") or {}).values()),
            "library_level1_count": len(lib_counts),
        },
    })


@app.route("/api/vocab", methods=["POST"])
def vocab_save():
    """保存编辑后的坑类型词表。body: {"by_level1": {"一级": ["二级", ...]}}
    另有 mode=merge_library 时，把案例库中实际用到的配对合并进来（不删除已有项）。"""
    data = request.get_json(force=True, silent=True) or {}
    mode = (data.get("mode") or "replace").strip()
    current = (load_pitfall_vocab() or {}).get("by_level1") or {}

    if mode == "merge_library":
        lib_pairs, _ = _vocab_library_pairs(load_cases())
        merged = {k: list(v) for k, v in current.items()}
        added = 0
        for k, vals in lib_pairs.items():
            bucket = merged.setdefault(k, [])
            for v in vals:
                if v not in bucket:
                    bucket.append(v)
                    added += 1
        saved = save_pitfall_vocab(merged)
        return jsonify({"ok": True, "added": added, "by_level1": saved["by_level1"],
                        "level1": saved["level1"], "level2": saved["level2"]})

    by_level1 = data.get("by_level1")
    if not isinstance(by_level1, dict):
        return jsonify({"ok": False, "error": "by_level1 必须是对象 {一级: [二级,...]}"}), 400
    saved = save_pitfall_vocab(by_level1)
    return jsonify({"ok": True, "by_level1": saved["by_level1"],
                    "level1": saved["level1"], "level2": saved["level2"]})


@app.route("/api/vocab/reset", methods=["POST"])
def vocab_reset():
    """从案例库重建词表（覆盖当前文件内容）。"""
    lib_pairs, _ = _vocab_library_pairs(load_cases())
    saved = save_pitfall_vocab(lib_pairs)
    return jsonify({"ok": True, "by_level1": saved["by_level1"],
                    "level1": saved["level1"], "level2": saved["level2"]})


# ---------------- 设置 ----------------
@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "GET":
        s = load_settings()
        safe = {k: v for k, v in s.items() if k not in ("admin_password_hash", "_secret_key")}
        return jsonify(safe)
    data = request.get_json(force=True, silent=True) or {}
    s = load_settings()
    for k in ("bot_api", "judge_model", "judge_base_url", "judge_api_key", "judge_temperature", "sample_times"):
        if k in data:
            s[k] = data[k]
    with _lock:
        save_settings(s)
    return jsonify({"ok": True, "settings": s})


# ---------------- 管理员：登录状态与密码 ----------------
@app.route("/api/auth/status")
def auth_status():
    s = load_settings()
    return jsonify({
        "ok": True,
        "password_set": bool(s.get("admin_password_hash")),
        "authenticated": bool(session.get("admin_authed")),
    })


@app.route("/api/login", methods=["POST"])
def login():
    s = load_settings()
    if not s.get("admin_password_hash"):
        return jsonify({"ok": False, "error": "尚未设置登录密码，无需登录"}), 400
    data = request.get_json(force=True, silent=True) or {}
    pw = (data.get("password") or "")
    if _verify_password(pw, s["admin_password_hash"]):
        session["admin_authed"] = True
        session.permanent = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "密码错误"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("admin_authed", None)
    return jsonify({"ok": True})


@app.route("/api/admin/set-password", methods=["POST"])
def admin_set_password():
    s = load_settings()
    data = request.get_json(force=True, silent=True) or {}
    cur = (data.get("current_password") or "")
    new = (data.get("new_password") or "")
    if len(new) < 4:
        return jsonify({"ok": False, "error": "新密码至少 4 位"}), 400
    existing = s.get("admin_password_hash") or ""
    if existing and not _verify_password(cur, existing):
        return jsonify({"ok": False, "error": "当前密码不正确"}), 401
    s["admin_password_hash"] = _hash_password(new)
    with _lock:
        save_settings(s)
    session["admin_authed"] = True
    session.permanent = True
    return jsonify({"ok": True, "password_set": True})


@app.route("/api/admin/remove-password", methods=["POST"])
def admin_remove_password():
    s = load_settings()
    data = request.get_json(force=True, silent=True) or {}
    pw = (data.get("password") or "")
    existing = s.get("admin_password_hash") or ""
    if not existing:
        return jsonify({"ok": False, "error": "当前未设置密码"}), 400
    if not _verify_password(pw, existing):
        return jsonify({"ok": False, "error": "密码不正确"}), 401
    s["admin_password_hash"] = ""
    with _lock:
        save_settings(s)
    session.pop("admin_authed", None)
    return jsonify({"ok": True, "password_set": False})


# ---------------- 裁判模型：测试连接 ----------------
@app.route("/api/judge/test", methods=["POST"])
def test_judge():
    """最小可用调用验证裁判模型 Key / endpoint / model 是否真实可用。
    优先用前端传入（非空），为空则回退已保存设置，便于「已保存 Key 但未填框」时直接测。"""
    data = request.get_json(force=True, silent=True) or {}
    s = load_settings()
    api_key = (data.get("judge_api_key") or "").strip() or (s.get("judge_api_key") or "")
    base_url = (data.get("judge_base_url") or "").strip() or (s.get("judge_base_url") or "https://api.openai.com/v1")
    model = (data.get("judge_model") or "").strip() or (s.get("judge_model") or "gpt-4o")

    if not api_key:
        return jsonify({"ok": False, "error": "未配置裁判 API Key（请先在输入框填写，或先在「设置」保存）"}), 400

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 1) 先校验鉴权与 endpoint（models.list 不涉及具体模型，最轻量）
        try:
            client.models.list()
        except Exception:
            pass  # 部分兼容网关不支持 /models，下面用真实对话兜底校验
        # 2) 真实极短对话：同时校验 token 有效、endpoint 可达、model 真实存在
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping（仅用于连接测试，请直接回复 OK）"}],
        )
        used = getattr(resp, "model", model)
        return jsonify({"ok": True, "model": used, "message": f"连接成功 · 模型 {used} 可用"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"连接失败：{e}"})


# ---------------- 引擎：TREK 评测 ----------------
@app.route("/api/evaluate/engineA", methods=["POST"])
def evaluate_engine_a():
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode", "gold")          # gold | mock | bot
    sample_size = int(data.get("sample_size", 20))
    full = bool(data.get("full", False))
    inject_defect = bool(data.get("inject_defect", False))
    mock_flaw = data.get("mock_flaw", "none")
    bot_struct_api = data.get("bot_struct_api", "") or ""
    case = data.get("case") or {}

    if not TREK_AVAILABLE:
        return jsonify({"ok": False, "error": "TREK 仓库不可用（缺 pandas/numpy 或 trek_repo）"}), 500

    try:
        per_task, agg = run_trek_engine(
            mode=mode, sample_size=sample_size, full=full,
            inject_defect=inject_defect, mock_flaw=mock_flaw,
            bot_struct_api=bot_struct_api,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"引擎 运行失败：{e}"}), 500

    tasks = []
    for (idx, meta, plan_output, sr, ev) in per_task:
        assertions = map_evidence_to_assertions(ev, meta, case)
        tasks.append({
            "task_index": idx,
            "impossible": getattr(meta, "impossible", False) if meta is not None else False,
            "query": getattr(meta, "query", "") if meta is not None else "",
            "assertions": assertions,
            "evidence": ev,
            "score": {
                "fully_valid": getattr(sr, "fully_valid", None) if sr else None,
                "task_success": getattr(sr, "task_success", None) if sr else None,
                "total_cost": getattr(sr, "total_cost", None) if sr else None,
            } if sr else None,
        })

    return jsonify({
        "ok": True,
        "mode": mode,
        "sample_size": len(per_task),
        "aggregate": agg,
        "tasks": tasks,
    })


# ---------------- 引擎 B：轻量裁判（补 TREK 盲区） ----------------
@app.route("/api/evaluate/engineB", methods=["POST"])
def evaluate_engine_b():
    data = request.get_json(force=True, silent=True) or {}
    case = data.get("case") or {}
    cfg = data.get("cfg") or {}
    try:
        res = run_engine_b(case, cfg)
    except Exception as e:
        return jsonify({"ok": False, "error": f"引擎 B 运行失败：{e}"}), 500
    return jsonify({"ok": True, "result": res})


# ----------------------------------------------------------------------------
# 避坑评测（融合三层）：对照题库底牌，评 Bot 行程是否避坑
# ----------------------------------------------------------------------------
@app.route("/api/evaluate/pitfall", methods=["POST"])
def evaluate_pitfall():
    data = request.get_json(force=True, silent=True) or {}
    case_id = data.get("case_id", "")
    mode = data.get("mode", "gold")  # gold / mock / bot
    cases = load_cases()
    case = next((c for c in cases if c.get("case_id") == case_id), None)
    if not case:
        return jsonify({"ok": False, "error": f"未找到案例 {case_id}"}), 404
    settings = load_settings()
    cfg = {
        "bot_api": data.get("bot_api", "") or settings.get("bot_api", ""),
        "judge_api_key": data.get("judge_api_key", "") or settings.get("judge_api_key", ""),
        "judge_model": data.get("judge_model", "") or settings.get("judge_model", ""),
        "judge_base_url": data.get("judge_base_url", "") or settings.get("judge_base_url", ""),
    }
    if mode not in ("gold", "mock", "bot"):
        mode = "gold"
    try:
        res = run_pitfall_eval(case, cfg, mode=mode, bot_api=cfg.get("bot_api"))
    except Exception as e:
        return jsonify({"ok": False, "error": f"避坑评测失败：{e}"}), 500
    return jsonify({"ok": True, "mode": mode, "case_id": case_id, "result": res,
                    "families": PITFALL_FAMILIES})


# ---------------- 单次案例双引擎报告 ----------------
@app.route("/api/evaluate/report", methods=["POST"])
def evaluate_report():
    data = request.get_json(force=True, silent=True) or {}
    case_id = data.get("case_id", "")
    cases = load_cases()
    case = next((c for c in cases if c.get("case_id") == case_id), None)
    if not case:
        return jsonify({"ok": False, "error": f"未找到案例 {case_id}"}), 404

    # 案例生成器：先把案例变成可执行的测试（用埋的坑造对抗输入、用正确应对造期望清单）
    try:
        generated_test = build_test_from_case(case)
    except Exception:
        generated_test = None

    settings = load_settings()
    engineA_source = data.get("engineA_source", "gold")
    engineA_flaw = data.get("engineA_flaw", "none")
    sample_size = int(data.get("sample_size", 20))

    a_mode = {"gold": "gold", "mock": "mock", "bot": "bot"}.get(engineA_source, "gold")
    bot_struct_api = data.get("bot_struct_api", "") or settings.get("bot_api", "")

    report = {"case": case, "engineA": None, "engineB": None}

    if TREK_AVAILABLE:
        try:
            per_task, agg = run_trek_engine(
                mode=a_mode, sample_size=sample_size, full=False,
                inject_defect=(engineA_flaw != "none"), mock_flaw=engineA_flaw,
                bot_struct_api=bot_struct_api,
            )
            tasks = []
            for (idx, meta, plan_output, sr, ev) in per_task:
                tasks.append({
                    "task_index": idx,
                    "impossible": getattr(meta, "impossible", False) if meta is not None else False,
                    "assertions": map_evidence_to_assertions(ev, meta, case),
                    "evidence": ev,
                    "score": {"fully_valid": getattr(sr, "fully_valid", None) if sr else None,
                              "total_cost": getattr(sr, "total_cost", None) if sr else None} if sr else None,
                })
            report["engineA"] = {"source": engineA_source, "aggregate": agg, "tasks": tasks, "sample_size": len(per_task)}
        except Exception as e:
            report["engineA"] = {"error": str(e)}

    # 引擎 B
    cfg = {
        "bot_api": data.get("bot_api", "") or settings.get("bot_api", ""),
        "judge_api_key": data.get("judge_api_key", "") or settings.get("judge_api_key", ""),
        "judge_model": data.get("judge_model", "") or settings.get("judge_model", "gpt-4o"),
        "judge_base_url": data.get("judge_base_url", "") or settings.get("judge_base_url", "https://api.openai.com/v1"),
    }
    try:
        report["engineB"] = run_engine_b(case, cfg, test=generated_test)
    except Exception as e:
        report["engineB"] = {"enabled": True, "error": str(e), "role": "lightweight_judge", "scope": "soft_quality_only"}
    report["generated_test"] = generated_test

    # 记录执行
    run_id = "RUN-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = load_runs()
    runs.insert(0, {
        "run_id": run_id,
        "trigger": "手动 · 生成器→TREK+轻量裁判",
        "case_id": case_id,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "engineA_source": engineA_source,
        "engineB_enabled": bool(cfg.get("judge_api_key") or settings.get("judge_api_key")),
    })
    save_runs(runs[:50])

    return jsonify({"ok": True, "run_id": run_id, "report": report})


# ---------------- 看板聚合 ----------------
@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    cases = load_cases()
    runs = load_runs()
    settings = load_settings()

    # 引擎 基准（gold）聚合作为硬逻辑得分率（真实、确定性）
    hard_rate = None
    if TREK_AVAILABLE:
        try:
            _, agg = run_trek_engine(mode="gold", sample_size=30, full=False)
            cat = agg.get("cat_satisfaction") or agg.get("cat_truthfulness")
            if isinstance(cat, (int, float)):
                hard_rate = round(cat * 100, 1)
        except Exception:
            hard_rate = None

    # 引擎 B 近期得分（从 runs 无法取具体分，这里用 cases.recent_score 聚合）
    scored = [c.get("recent_score") for c in cases if isinstance(c.get("recent_score"), (int, float))]
    soft_avg = round(sum(scored) / len(scored), 1) if scored else None

    return jsonify({
        "ok": True,
        "case_count": len(cases),
        "run_count": len(runs),
        "hard_logic_rate": hard_rate,
        "soft_avg": soft_avg,
        "trek_available": TREK_AVAILABLE,
        "judge_configured": bool(settings.get("judge_api_key")),
        "bot_configured": bool(settings.get("bot_api")),
        "runs": runs[:8],
        "cases": cases,
    })


@app.route("/api/runs", methods=["GET"])
def runs():
    return jsonify({"ok": True, "runs": load_runs()})


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("旅游 AI 质检平台后端启动 → http://localhost:8501")
    print(f"TREK_AVAILABLE={TREK_AVAILABLE}")
    app.run(host="0.0.0.0", port=8501, debug=False, threaded=True)
