# -*- coding: utf-8 -*-
"""
一次性导入：把《需求测试题库.xlsx》主表（带坑需求测试题库）的 1000 行
转换为新案例模型，写入 cases.json；并迁移原有 3 条业务案例到新模型。
仅运行一次，可重复运行（覆盖式重建 cases.json）。
"""
import os, json, re
import openpyxl

BASE = os.path.dirname(os.path.abspath(__file__))
XLSX = r"C:\Users\84288\Downloads\需求测试题库.xlsx"
OUT = os.path.join(BASE, "cases.json")

# 19 个高频一级坑类型（用于把自由文本坑类型归并到一级分类）
LEVEL1_KNOWN = [
    "机票", "签证", "酒店", "儿童", "自驾", "门票", "季节", "安全", "交通",
    "健康", "季节天气", "驾照", "费用", "高原", "宗教着装", "气候", "宗教",
    "行程", "着装",
]

def parse_pitfall_types(raw):
    """把「门票+季节+机票+酒店」「交通(ZTL)+酒店」「酒店(年龄)+签证」解析为两级标签。"""
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
    # 把一级归并到已知一级（模糊匹配，避免散碎标签）
    for it in out:
        l1 = it["level1"]
        hit = next((k for k in LEVEL1_KNOWN if k == l1 or k in l1 or l1 in k), None)
        if hit:
            it["level1"] = hit
    return out

def split_service_types(raw):
    if not raw or not str(raw).strip():
        return []
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

def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    ws = wb["带坑需求测试题库"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    idx = {name: i for i, name in enumerate(header)}

    def col(name):
        return idx.get(name, -1)

    cases = []
    for r in rows[1:]:
        if r is None:
            continue
        def get(name):
            i = col(name)
            return r[i] if (i >= 0 and i < len(r)) else None
        case_id = str(get("编号") or "").strip()
        if not case_id:
            continue
        scene = str(get("场景") or "").strip()
        dest, period = derive_destination_period(scene)
        pit = parse_pitfall_types(get("坑类型"))
        cases.append({
            "case_id": case_id,
            "scene": scene,
            "destination": dest,
            "travel_period": period,
            "user_prompt": str(get("客户原话(测试输入)") or "").strip(),
            "travel_time": str(get("出行时间") or "").strip(),
            "group_size": str(get("人数") or "").strip(),
            "service_types": split_service_types(get("服务类型")),
            "pitfall_types": pit,
            "buried_pitfalls": str(get("埋的坑") or get("埋的坑(评分用·不喂AI)") or "").strip(),
            "correct_response": str(get("正确应对") or get("正确应对(评分用·不喂AI)") or "").strip(),
            "category": dest or "未分类",
            "hard_assertions": {"max_budget": None, "max_daily_walking_km": None},
            "known_pitfalls": sorted({p["level1"] for p in pit}),
            "source": "需求测试题库",
            "gen_method": "上传导入",
            "zone": "A+B",
            "recent_score": None,
        })

    # 迁移原有 3 条业务案例到新模型（保留 hard_assertions 与 known_pitfalls）
    legacy = [
        {"case_id": "TC_XIAN_001", "category": "西安·历史", "destination": "西安",
         "user_prompt": "我们下个月准备去西安玩3天2晚，主要是想看历史文化景点，带着爸妈一起，希望节奏别太赶，预算大概每人3000块，帮忙给个方案。",
         "hard_assertions": {"max_budget": 6000, "max_daily_walking_km": 4.0},
         "known_pitfalls": ["陕西历史博物馆周一闭馆且极难预约", "中老年人单日暴走强度过高"]},
        {"case_id": "TC_JAPAN_KANSAI_001", "category": "日本关西", "destination": "日本关西",
         "user_prompt": "带老人和孩子去日本关西7天，大阪进京都出，想看清水寺、环球影城，预算人均1.5万，行程不要太累。",
         "hard_assertions": {"max_budget": 30000, "max_daily_walking_km": 6.0},
         "known_pitfalls": ["环球影城需提前买快速通行", "关西机场到大阪市区可行李配送"]},
        {"case_id": "TC_EUROPE_FAMILY_001", "category": "欧洲亲子", "destination": "欧洲",
         "user_prompt": "一家三口去法国+瑞士10天，巴黎进苏黎世出，想看法艺术博物馆也看雪山，预算人均3万。",
         "hard_assertions": {"max_budget": 90000, "max_daily_walking_km": 8.0},
         "known_pitfalls": ["申根签证需提前预约", "瑞士公共交通通票划算"]},
    ]
    for lg in legacy:
        cases.append({
            "case_id": lg["case_id"],
            "scene": lg.get("category", ""),
            "destination": lg.get("destination", lg.get("category", "")),
            "travel_period": "",
            "user_prompt": lg["user_prompt"],
            "travel_time": "",
            "group_size": "",
            "service_types": [],
            "pitfall_types": [{"level1": p.split("需")[0].split("周一")[0][:2], "level2": ""} for p in lg["known_pitfalls"]][:1] if lg["known_pitfalls"] else [],
            "buried_pitfalls": "；".join(lg["known_pitfalls"]),
            "correct_response": "",
            "category": lg["category"],
            "hard_assertions": lg["hard_assertions"],
            "known_pitfalls": lg["known_pitfalls"],
            "source": "business",
            "gen_method": "手动新增",
            "zone": "A+B",
            "recent_score": None,
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    # 统计
    dests = sorted({c["destination"] for c in cases if c["destination"]})
    svc = sorted({s for c in cases for s in c["service_types"]})
    l1 = sorted({p["level1"] for c in cases for p in c["pitfall_types"]})
    print("导入完成：", len(cases), "条")
    print("目的地去重：", len(dests))
    print("服务类型原子项：", len(svc))
    print("坑类型一级：", len(l1), "->", l1)

if __name__ == "__main__":
    main()
