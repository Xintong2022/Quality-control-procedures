#!/usr/bin/env python3
"""
Verify the *achievable ceiling*: score the human-verified gold references and confirm that
every task reaches a perfect 1.0 under the deterministic evaluator.

This is the paper's central reproducibility claim, and it runs **fully offline** --- no API
keys, no network, no LLM --- using only `score_trek.py`, the knowledge base, and `trek_gold.jsonl`.

    python verify_gold.py
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    gold_path = os.path.join(ROOT, "trek_gold.jsonl")
    records = [json.loads(line) for line in open(gold_path, encoding="utf-8") if line.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        in_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        # Present the gold as an agent-results file: mark it submitted so the scorer credits refusals.
        with open(os.path.join(in_dir, "trek_gold.jsonl"), "w", encoding="utf-8") as f:
            for i, rec in enumerate(records):
                rec = dict(rec)
                rec["submitted"] = True
                rec["query_index"] = i
                rec["tool_call_count"] = 0
                f.write(json.dumps(rec) + "\n")

        subprocess.run(
            [sys.executable, os.path.join(ROOT, "score_trek.py"),
             "--results-dir", in_dir, "--out-dir", out_dir],
            check=True,
        )
        summary = json.load(open(os.path.join(out_dir, "summary.json"), encoding="utf-8"))

    m = summary[0] if isinstance(summary, list) else summary
    tp_feas = m.get("task_perfect_feasible")
    tp_inf = m.get("task_perfect_infeasible")
    print(f"\ngold task-perfect rate:  feasible = {tp_feas}   infeasible = {tp_inf}")
    ok = tp_feas == 1.0 and tp_inf == 1.0
    print("PASS — the ceiling is demonstrably reachable: every gold task scores 1.0."
          if ok else "FAIL — a gold task did not reach 1.0; see the printed scores above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
