# -*- coding: utf-8 -*-
"""
一次性 / 可续跑：把 cases.json 每个案例的「正确应对 + 埋的坑」拆成结构化避坑断言，
写入案例的 pitfall_assertions 字段。供 pitfall_engine 做逐坑命中评测。

用法（pilot 先验证质量）：
  python build_assertions.py --limit 12
  python build_assertions.py --idx 1,7,15
全量（约 1000 条，注意模型调用量与耗时）：
  python build_assertions.py --all
续跑（跳过已有断言的案例）：
  python build_assertions.py --all --resume
"""
import os
import sys
import json
import argparse
import time

BASE = os.path.dirname(os.path.abspath(__file__))
CASES_FILE = os.path.join(BASE, "cases.json")
sys.path.insert(0, BASE)
from pitfall_engine import FAMILIES, FAMILY_GUIDE

from openai import OpenAI


def load_settings():
    p = os.path.join(BASE, "settings.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


SYS_PROMPT = (
    "你是旅游质检平台的「避坑断言分解器」。给定一条测试案例的『埋的坑』(客户易踩、Bot 易翻车的真实旅行风险) "
    "与『正确应对』(优秀顾问应当做到的具体动作)，请把『正确应对』拆成若干条**原子、可独立判定**的避坑断言。\n"
    "每条断言要求：\n"
    "  - text：一句具体动作（如『提示申根保险需≥3万欧并覆盖全程含医疗运送』），不要泛泛而谈。\n"
    "  - dimension_family：从 8 个维度族中选最贴切的一个。\n"
    "  - fatal：是否『致命红线』——若 Bot 遗漏此项会直接导致安全风险/违法/无法入境/行程作废，则为 true；否则 false。\n"
    "8 个维度族说明：" + FAMILY_GUIDE + "\n"
    "严格只输出 JSON：{\"assertions\":[{\"text\":<str>,\"dimension_family\":<str>,\"fatal\":<bool>}]}\n"
    "要求：覆盖正确应对的关键点；每条断言只表达一个动作；致命项宁少勿滥。"
)


def build_user(case):
    buried = (case.get("buried_pitfalls") or "").strip()
    correct = (case.get("correct_response") or "").strip()
    return (
        f"场景：{case.get('scene','')}（{case.get('destination','')} {case.get('travel_period','')}）\n"
        f"客户原话：{case.get('user_prompt','')}\n"
        f"【埋的坑·应规避】\n{buried or '（无）'}\n"
        f"【正确应对·期望做到】\n{correct or '（无）'}\n"
    )


def decompose(case, cfg):
    key = (cfg or {}).get("judge_api_key") or ""
    base = (cfg or {}).get("judge_base_url") or "https://api.openai.com/v1"
    model = (cfg or {}).get("generator_model") or (cfg or {}).get("judge_model") or "gpt-4o"
    if not key:
        return None, "未配置裁判/生成 Key"
    client = OpenAI(api_key=key, base_url=base)
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": build_user(case)},
            ],
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
        arr = obj.get("assertions") or []
        out = []
        for a in arr:
            if not isinstance(a, dict):
                continue
            txt = (a.get("text") or "").strip()
            if not txt:
                continue
            fam = a.get("dimension_family") or ""
            if fam not in FAMILIES:
                fam = "通用适配"
            out.append({"text": txt, "dimension_family": fam, "fatal": bool(a.get("fatal", False))})
        return out, None
    except Exception as e:
        return None, f"分解失败：{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（默认 0=不限制）")
    ap.add_argument("--idx", type=str, default="", help="指定案例编号，逗号分隔，如 1,7,15")
    ap.add_argument("--all", action="store_true", help="处理全部案例")
    ap.add_argument("--resume", action="store_true", help="跳过已有 pitfall_assertions 的案例")
    ap.add_argument("--dry", action="store_true", help="只打印将处理哪些，不调用模型")
    args = ap.parse_args()

    cfg = load_settings()
    cases = json.load(open(CASES_FILE, encoding="utf-8"))

    targets = []
    if args.idx:
        want = set(x.strip() for x in args.idx.split(",") if x.strip())
        targets = [c for c in cases if str(c.get("case_id", "")) in want]
    else:
        for c in cases:
            if args.resume and c.get("pitfall_assertions"):
                continue
            if not args.all and not args.limit:
                continue
            targets.append(c)
        if args.limit:
            targets = targets[:args.limit]

    print(f"待处理案例数：{len(targets)}（全部 {len(cases)}）")
    if args.dry:
        for c in targets[:30]:
            print("  ", c.get("case_id"), c.get("destination"), "|", (c.get("correct_response") or "")[:30])
        return

    done = 0
    for c in targets:
        cid = c.get("case_id")
        asr, err = decompose(c, cfg)
        if err:
            print(f"  [跳过] {cid}: {err}")
            continue
        c["pitfall_assertions"] = asr
        done += 1
        fams = {}
        for a in asr:
            fams[a["dimension_family"]] = fams.get(a["dimension_family"], 0) + 1
        print(f"  [OK] {cid}: {len(asr)} 条断言 | {fams}")
        time.sleep(0.3)

    json.dump(cases, open(CASES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n完成：成功分解 {done} 条，已写回 {CASES_FILE}")


if __name__ == "__main__":
    main()
