#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock TREK Bot —— 方案 1（让 Bot 直接输出 TREK 结构化计划）的参考实现 / 合同校验服务。

这是一个**零额外依赖**的 HTTP 服务（仅用 Python 标准库），用于：
  1. 让你在不接入真实旅游 Bot 的情况下，端到端验证「引擎 直接对结构化计划打分」的链路；
  2. 作为你改造真实 Bot 时的**合同范本**——你的 Bot 只要按同样的请求/响应格式实现，
     引擎 就能确定性地直接打分。

运行：
    python mock_trek_bot.py            # 监听 http://localhost:8601
    python mock_trek_bot.py --port 8602

请求（POST /generate，application/json）：
    {
      "query": "<TREK 查询文本>",
      "query_index": 12,
      "budget": 2000.0,
      "days": 3,
      "cities_count": 1,
      "impossible": false,
      "flaw": "none"          # 可选: none | omit_attraction | fake_attraction | wrong_refusal
    }

响应（JSON，即 TREK 结构化计划）：
    {
      "is_feasible": true,
      "refusal_reason": "",
      "plan": { "day1": {...}, ... }
    }

引擎 对返回的计划做确定性评分（预算/实体真实性/城市约束/不可行拒绝/营业时间/时空可达 等 10 维）。
"""
import argparse
import copy
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD_PATH = os.path.join(HERE, "trek_repo", "trek_gold.jsonl")

FLAW_NONE = "none"
FLAW_OMIT = "omit_attraction"
FLAW_FAKE = "fake_attraction"
FLAW_REFUSE = "wrong_refusal"


def load_gold():
    recs = []
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


GOLD = load_gold()


def apply_flaw(plan_output, flaw, meta):
    plan = plan_output.get("plan", {})
    if not isinstance(plan, dict):
        return
    if flaw == FLAW_REFUSE and not meta.get("impossible"):
        plan_output["is_feasible"] = False
        plan_output["refusal_reason"] = "（Mock 注入）抱歉，我无法为您规划此行程"
        plan_output["plan"] = {}
        return
    if flaw == FLAW_NONE:
        return
    for day_key in sorted(plan.keys(), key=str):
        day = plan.get(day_key)
        if not isinstance(day, dict):
            continue
        attrs = day.get("attractions") or day.get("attraction")
        if isinstance(attrs, list) and attrs:
            if flaw == FLAW_OMIT:
                attrs.pop()
                return
            if flaw == FLAW_FAKE and isinstance(attrs[0], dict):
                attrs[0] = dict(attrs[0])
                attrs[0]["name"] = "【虚构】不存在的梦幻之地XYZ"
                return


def build_response(payload):
    idx = int(payload.get("query_index", 0))
    rec = GOLD[idx] if 0 <= idx < len(GOLD) else None
    if rec is None:
        return {"is_feasible": False, "refusal_reason": "未知 query_index", "plan": {}}
    out = {
        "is_feasible": rec.get("is_feasible", False),
        "refusal_reason": rec.get("refusal_reason", ""),
        "plan": copy.deepcopy(rec.get("plan", {})),
    }
    apply_flaw(out, payload.get("flaw", FLAW_NONE), payload)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/") not in ("/generate", ""):
            self._send({"error": "only POST /generate supported"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            self._send({"error": f"bad request: {e}"}, 400)
            return
        try:
            self._send(build_response(payload))
        except Exception as e:
            self._send({"error": str(e)}, 500)

    def do_GET(self):
        self._send({
            "service": "mock_trek_bot",
            "endpoints": ["POST /generate"],
            "gold_tasks": len(GOLD),
            "note": "返回 TREK Gold 结构化计划，支持 flaw 参数注入缺陷，用作引擎 合同校验。",
        })

    def log_message(self, fmt, *args):
        print("[mock_trek_bot] " + (fmt % args))


def main():
    ap = argparse.ArgumentParser(description="Mock TREK Bot 参考实现")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8601)
    args = ap.parse_args()
    print(f"Mock TREK Bot 启动: http://{args.host}:{args.port}/generate  (gold tasks={len(GOLD)})")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
