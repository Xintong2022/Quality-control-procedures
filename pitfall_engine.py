# -*- coding: utf-8 -*-
"""
避坑评测引擎（融合三层）

背景：原 TREK 引擎只评"行程机械合法性"（实体真实/预算/城市约束/营业时间…），
对照的是合成知识库，完全没碰题库里真实的"坑"。本引擎改为拿 Bot 的行程/回答，
对照题库的「正确应对」底牌，评它"有没有避坑"。

三层融合：
  Layer 0  红线零容忍：fatal=true 的断言若被遗漏 → 该案例直接判 F（一票否决）
  Layer 1  分层维度卡：按 8 大避坑维度族算逐维命中率 + 加权总分
  Layer 2  LLM 语义裁判：对每条断言判 命中/遗漏 + 证据 + 置信度（补深度）

依赖：OpenAI 兼容裁判客户端（复用 settings 的 judge_* 配置）。
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# 8 大避坑维度族（直接源自题库真实坑类型分布：机票/签证/酒店/儿童/自驾/门票/安全…）
# ----------------------------------------------------------------------------
FAMILIES = [
    "证件合规",      # 签证/护照/保险/驾照翻译/ETA/黄皮书/海关/邀请函/未成年授权
    "时间窗口",      # 闭馆/维修/登山季封闭/台风/观鲸/迁徙/花期/极光/旺季人潮
    "预订限流",      # 吉卜力/美泉宫预约/限流/车证/快速通/水屋/营地/整理券
    "交规地方法规",  # ZTL/右舵/雪胎/儿童座椅/禁车村/高速贴纸/非联程行李/拥堵费
    "安全健康",      # 高原/热气球资质/潜水年龄/治安扒手/政局/水母/疫苗/风暴
    "地理动线",      # 北岛南岛矛盾/双机场/接机宵禁/岛际飞/偏远/摆渡/转机
    "费用陷阱",      # 圣诞高价/罚单滞后/年轻司机费/取消政策/双货币/小费/环保费
    "人群适配",      # 老人暴走/儿童年龄座椅/宗教着装/未成年人入住/带娃体验
]

FAMILY_GUIDE = (
    "证件合规（签证/护照/保险/驾照翻译/ETA/黄皮书/海关申报/邀请函/未成年授权书等合规文件）；"
    "时间窗口（闭馆/维修/登山季封闭/台风飓风/观鲸季/动物迁徙/花期/极光/旺季人潮/高温等季节与开放时间约束）；"
    "预订限流（景点预约/闭馆日/限流小镇/车证/快速通行/水屋/营地/整理券/入堡时段等需提前订或限流）；"
    "交规地方法规（ZTL限行/右舵靠左/雪胎强制/儿童座椅法/禁车村接驳/高速贴纸/非联程行李不直挂/拥堵费/山路四驱等当地交规）；"
    "安全健康（高原适应/热气球资质/潜水年龄下限/扒手治安/政局局势/水母珊瑚/疫苗黄皮书/风暴气旋/船证等安全健康风险）；"
    "地理动线（北岛南岛矛盾/双机场选错/接机宵禁/岛际飞行/偏远无路/摆渡接驳/转机链路等地理与动线合理性）；"
    "费用陷阱（圣诞新年高价/罚单滞后/年轻司机费/取消政策/双货币/小费/换汇/国家公园费/环保费等隐性费用）；"
    "人群适配（老人暴走强度/儿童年龄与座椅/宗教着装要求/未成年人入住/带娃体验差等人群特殊适配）。"
)

# 兜底：从坑类型一级关键词推测维度族（仅用于无结构化断言时的 fallback）
_FAMILY_KEYWORDS = {
    "证件合规": ["签证", "护照", "保险", "驾照", "ETA", "黄皮", "海关", "邀请函", "授权", "材料", "申报", "落地签", "电子签", "免签"],
    "时间窗口": ["季节", "天气", "气候", "闭馆", "花期", "台风", "飓风", "极光", "迁徙", "旺季", "高温", "寒假", "暑假", "雨季", "季风", "雪", "极夜"],
    "预订限流": ["门票", "预约", "限流", "车证", "快速", "水屋", "营地", "整理券", "入堡", "夜场", "钟楼", "城堡", "时段"],
    "交规地方法规": ["自驾", "ZTL", "右舵", "雪胎", "儿童座椅", "禁车", "高速贴纸", "拥堵费", "行李", "中转", "机场", "接机", "山路", "四驱", "越野", "靠左", "封路"],
    "安全健康": ["安全", "健康", "高原", "热气球", "潜水", "水母", "治安", "扒手", "政局", "动物", "地震", "火山", "风暴", "疫苗", "黄热", "船证", "风浪"],
    "地理动线": ["交通", "岛际", "渡轮", "火车", "内陆飞", "转机", "偏远", "地理", "双机场", "摆渡", "接驳"],
    "费用陷阱": ["费用", "取消", "罚单", "司机费", "货币", "小费", "换汇", "公园费", "环保费", "昂贵"],
    "人群适配": ["儿童", "老人", "宗教", "着装", "未成年", "带娃", "亲子", "年龄", "楼梯", "身高"],
}


def guess_family(text):
    """关键词兜底推测维度族；命中多个取首个，都不中回『通用适配』。"""
    for fam, kws in _FAMILY_KEYWORDS.items():
        for kw in kws:
            if kw in (text or ""):
                return fam
    return "通用适配"


def get_assertions(case):
    """取案例的避坑断言：优先结构化 pitfall_assertions；否则从正确应对兜底派生。"""
    pa = case.get("pitfall_assertions")
    if isinstance(pa, list) and pa:
        out = []
        for a in pa:
            if not isinstance(a, dict):
                continue
            txt = (a.get("text") or "").strip()
            if not txt:
                continue
            fam = a.get("dimension_family") or guess_family(txt)
            out.append({
                "text": txt,
                "dimension_family": fam if fam in FAMILIES else guess_family(txt),
                "fatal": bool(a.get("fatal", False)),
            })
        if out:
            return out
    # 兜底：把 correct_response 的句子直接当断言
    correct = (case.get("correct_response") or "").strip()
    if correct:
        from server import _split_sentences
        return [{"text": s, "dimension_family": guess_family(s), "fatal": False}
                for s in _split_sentences(correct) if s.strip()]
    return []


# ----------------------------------------------------------------------------
# LLM 语义裁判（Layer 2）：一次调用判完一个案例的全部断言
# ----------------------------------------------------------------------------
def _judge_client(cfg):
    from openai import OpenAI
    key = (cfg or {}).get("judge_api_key") or ""
    base = (cfg or {}).get("judge_base_url") or "https://api.openai.com/v1"
    return OpenAI(api_key=key, base_url=base), key


def judge_assertions(case, assertions, bot_text, cfg):
    """对每条断言，让裁判 LLM 判 命中/遗漏 + 证据 + 置信度。

    返回 list[{text, dimension_family, fatal, hit(bool), evidence, confidence}]。
    无裁判 Key 时退化为"无法评判"占位（hit=None）。
    """
    if not assertions:
        return []
    client, key = _judge_client(cfg)
    if not key:
        return [dict(a, hit=None, evidence="未配置裁判 Key，无法语义裁判", confidence=0.0)
                for a in assertions]

    items = "\n".join(
        f"{i+1}. [维度={a['dimension_family']}|致命={a['fatal']}] {a['text']}"
        for i, a in enumerate(assertions)
    )
    sys_prompt = (
        "你是旅游质检的「避坑语义裁判」。给定一条测试案例的 Bot 回答，以及一组「正确应对断言」"
        "（即优秀顾问应当做到、以规避真实旅行坑点的具体动作）。\n"
        "请逐条判断：Bot 回答是否**实际包含并执行/提示**了该断言所述动作。\n"
        "判定口径：\n"
        "  - hit=true 仅当 Bot 明确做到了该动作（含等价表述/主动提示/替代方案）；泛泛承诺、未提及、做反了都算 hit=false。\n"
        "  - evidence 摘录 Bot 回答中支持你判定的原句（≤40字）；若 hit=false 写『未提及/做反』。\n"
        "  - confidence 为 0~1，表示你判定的把握。\n"
        "严格只输出 JSON：{\"items\":[{\"idx\":<int>,\"hit\":<bool>,\"evidence\":<str>,\"confidence\":<0-1>}]}"
    )
    user_prompt = (
        f"原始用户需求：{case.get('user_prompt','')}\n\n"
        f"Bot 回答：\n{bot_text}\n\n"
        f"正确应对断言（逐条判定）：\n{items}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=(cfg or {}).get("judge_model") or "gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        judged = {}
        for it in (parsed.get("items") or []):
            try:
                idx = int(it.get("idx")) - 1
            except Exception:
                continue
            if 0 <= idx < len(assertions):
                judged[idx] = it
        out = []
        for i, a in enumerate(assertions):
            j = judged.get(i, {})
            out.append({
                "text": a["text"],
                "dimension_family": a["dimension_family"],
                "fatal": a["fatal"],
                "hit": bool(j.get("hit", False)),
                "evidence": str(j.get("evidence", "")),
                "confidence": float(j.get("confidence", 0.0) or 0.0),
            })
        return out
    except Exception as e:
        return [dict(a, hit=None, evidence=f"裁判调用失败：{e}", confidence=0.0)
                for a in assertions]


# ----------------------------------------------------------------------------
# 聚合（Layer 1）：逐维命中率 + 加权总分 + 红线零容忍
# ----------------------------------------------------------------------------
def aggregate(judged):
    fam_stats = {f: {"total": 0, "hit": 0, "fatal_total": 0, "fatal_miss": 0} for f in FAMILIES}
    n = len(judged)
    hits = 0
    fatal_miss = 0
    judged_n = 0
    for a in judged:
        if a["hit"] is None:
            continue
        judged_n += 1
        fam = a["dimension_family"]
        if fam not in fam_stats:
            fam = "通用适配"
            fam_stats.setdefault(fam, {"total": 0, "hit": 0, "fatal_total": 0, "fatal_miss": 0})
        fam_stats[fam]["total"] += 1
        if a["hit"]:
            fam_stats[fam]["hit"] += 1
            hits += 1
        if a["fatal"]:
            fam_stats[fam]["fatal_total"] += 1
            if not a["hit"]:
                fam_stats[fam]["fatal_miss"] += 1
                fatal_miss += 1

    # 逐维命中率
    family_rates = {}
    for f, s in fam_stats.items():
        if s["total"]:
            family_rates[f] = round(s["hit"] / s["total"], 3)
    # 加权总分（致命项权重翻倍；命中率口径）
    total_w = 0.0
    hit_w = 0.0
    for f, s in fam_stats.items():
        if not s["total"]:
            continue
        w = s["total"] + s["fatal_total"]  # 致命项额外加权
        total_w += w
        hit_w += w * (s["hit"] / s["total"])
    overall = round(hit_w / total_w, 3) if total_w else None

    return {
        "overall": overall,                 # 0~1 避坑得分
        "hit_rate": round(hits / judged_n, 3) if judged_n else None,
        "assertions_total": n,
        "assertions_judged": judged_n,
        "fatal_missed": fatal_miss,
        "fatal_zero_tolerance": fatal_miss > 0,  # 红线触发
        "family_rates": family_rates,
        "family_stats": fam_stats,
    }


def run_pitfall_eval(case, cfg, bot_text=None, mode="bot", bot_api=None):
    """编排一次避坑评测。

    mode:
      - "gold": Bot 完美回答（覆盖全部断言）→ 自检管线，期望 ~1.0
      - "mock": 完美回答但故意漏掉 1 条非致命断言 → 演示能抓遗漏
      - "bot" : 调真实 Bot（bot_api + 生成器测试输入）拿回答
    返回 {mode, assertions(judged), aggregate, bot_text, error}
    """
    assertions = get_assertions(case)
    if not assertions:
        return {"mode": mode, "assertions": [], "aggregate": None,
                "bot_text": "", "error": "该案例无正确应对断言，无法评避坑"}

    if bot_text is None:
        if mode == "gold":
            bot_text = "（完美参考回答）" + "；".join(a["text"] for a in assertions)
        elif mode == "mock":
            drop = next((a for a in assertions if not a["fatal"]), None)
            keep = [a["text"] for a in assertions if a is not drop]
            bot_text = "（含缺陷的参考回答）" + "；".join(keep)
            if drop:
                bot_text += f"【此处故意遗漏：{drop['text']}】"
        else:  # bot
            from server import build_test_from_case
            test = build_test_from_case(case)
            query = test.get("generated_query") or case.get("user_prompt", "")
            bot_text = ""
            api = bot_api or (cfg or {}).get("bot_api") or ""
            if api:
                try:
                    import requests
                    r = requests.post(api, json={"query": query, "case_id": case.get("case_id")}, timeout=30)
                    r.raise_for_status()
                    d = r.json()
                    bot_text = d.get("bot_text") or d.get("answer") or d.get("text") or ""
                    if not isinstance(bot_text, str):
                        bot_text = json.dumps(bot_text, ensure_ascii=False)
                except Exception as e:
                    bot_text = f"（Bot 调用失败：{e}）"
            else:
                bot_text = "（未配置 Bot API，以下为模拟回答占位）" + query

    if mode in ("gold", "mock"):
        # 确定性基线：gold 全命中；mock 漏掉故意 drop 的那条（不消耗 LLM，可做管线自检/演示）
        drop = next((a for a in assertions if not a["fatal"]), None) if mode == "mock" else None
        judged = []
        for a in assertions:
            if mode == "mock" and a is drop:
                judged.append(dict(a, hit=False, evidence="（mock 注入缺陷）故意遗漏", confidence=1.0))
            else:
                judged.append(dict(a, hit=True, evidence="（gold 基线）参考回答已覆盖", confidence=1.0))
    else:
        judged = judge_assertions(case, assertions, bot_text, cfg)
    agg = aggregate(judged)
    return {"mode": mode, "assertions": judged, "aggregate": agg, "bot_text": bot_text, "error": None}
