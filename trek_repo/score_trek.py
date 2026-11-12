#!/usr/bin/env python3
"""
Score TREK runs produced by run_trek.py.

Reads per-model JSONL from --results-dir (trek_<label>.jsonl, one agent result per line, each with
query_index / plan / is_feasible / refusal_reason / tool_call_count), aligns each row to
trek_queries.csv by query_index, scores it with the SAME TravelPlanScorer the benchmark defines, and
writes a per-query detail CSV plus one summary row per model (all dimensions + the 4 category scores +
weighted overall).

  python score_trek.py --results-dir trek_results --meta trek_queries.csv --out-dir trek_scores

No faiss/embeddings needed: D1 is deterministic set-intersection and D0-src verifies against the KB
CSVs, so scoring runs anywhere pandas + the v2 data are present.
"""
import argparse
import csv
import glob
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_queries, get_sandbox_db          # noqa: E402
from scoring import (TravelPlanScorer, compute_aggregate_scores,
                     bootstrap_aggregate, paired_bootstrap_diff)  # noqa: E402

DIMS = ["d0_keyword", "d0_source", "d1_implicit", "d2_unord_single", "d2_unord_multi",
        "d3_budget", "d4_impossible", "d5_retry", "b2_opening_hours", "b3_spatiotemporal"]
CATS = ["cat_satisfaction", "cat_truthfulness", "cat_reasoning", "cat_infeasibility",
        "cat_efficiency", "weighted_overall", "overall_avg",
        "fully_valid_rate", "avg_total_tokens", "avg_elapsed_sec"]


def as_plan_output(rec: dict) -> dict:
    """Rebuild the {is_feasible, plan, refusal_reason} payload the scorer's extract_plan_data reads.
    run_trek.py stores these at top level; tolerate an already-nested plan_output too."""
    plan = rec.get("plan", {})
    if isinstance(plan, dict) and ("is_feasible" in plan or "plan" in plan):
        return plan  # already a full payload
    return {
        "is_feasible": rec.get("is_feasible", False),
        "plan": plan if isinstance(plan, dict) else {},
        "refusal_reason": rec.get("refusal_reason", ""),
    }


def read_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def score_file(path: str, queries, scorer, out_dir: str) -> dict:
    label = os.path.splitext(os.path.basename(path))[0]
    records = read_jsonl(path)
    # A resume appends a retry line for a previously-errored query_index; keep the LAST occurrence.
    by_idx = {}
    for i, rec in enumerate(records):
        by_idx[rec.get("query_index", i)] = rec

    # SCORE THE WHOLE BENCHMARK, NOT THE ROWS THAT HAPPENED TO SURVIVE.
    # Aggregating over whatever is in the JSONL let a crashed or rate-limited run be graded on a
    # self-selected easy subset: dropping the 160 worst rows of a fixed model raised its overall
    # from 0.6464 to 0.8055. A query with no result is a failed attempt, so it is scored as an
    # empty submission — the denominator is the benchmark's, identical for every model.
    results = []
    n_err = 0
    n_missing = 0
    for idx in range(len(queries)):
        meta = queries[idx]
        rec = by_idx.get(idx)
        if rec is None:
            n_missing += 1
            rec = {"is_feasible": False, "plan": {}, "refusal_reason": "",
                   "tool_call_count": 0, "submitted": False}
        try:
            sr = scorer.score_query(
                as_plan_output(rec), meta, rec.get("tool_call_count", 0),
                total_tokens=(rec.get("token_usage") or {}).get("total_tokens"),
                elapsed_sec=rec.get("elapsed_sec"))
            sr.query_index = idx
            # A crashed / un-submitted run must NOT be scored as a correct refusal: it never chose
            # to refuse. Zero D4 for impossible tasks where the agent didn't actually submit.
            submitted_ok = bool(rec.get("submitted")) and not rec.get("error")
            if getattr(meta, "impossible", False) and not submitted_ok and sr.d4_impossible is not None:
                sr.d4_impossible = 0.0
            results.append(sr)
        except Exception as e:
            # A plan the scorer cannot parse (e.g. a model nested {"car": {"car_type": {..}}}) is a
            # FAILED submission, not a non-existent task. Dropping it broke the scorer's own
            # identical-denominator guarantee (gemma was silently scored on 522 feasible tasks, not
            # 533) and rewarded the most-malformed model. Re-score it as an empty submission — which
            # cannot crash and yields the correct all-fail row — so it stays in the denominator.
            n_err += 1
            print(f"  [warn] {label} q{idx}: {type(e).__name__}: {e} -> scored as failed submission")
            sr = scorer.score_query({"is_feasible": False, "plan": {}, "refusal_reason": ""},
                                    meta, rec.get("tool_call_count", 0),
                                    total_tokens=(rec.get("token_usage") or {}).get("total_tokens"),
                                    elapsed_sec=rec.get("elapsed_sec"))
            sr.query_index = idx
            if getattr(meta, "impossible", False) and sr.d4_impossible is not None:
                sr.d4_impossible = 0.0
            results.append(sr)

    # per-query detail CSV
    detail_path = os.path.join(out_dir, f"{label}.scores.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_index"] + DIMS + ["is_feasible", "total_cost", "attraction_count"])
        for r in sorted(results, key=lambda x: x.query_index):
            w.writerow([r.query_index] + [getattr(r, d) for d in DIMS]
                       + [r.is_feasible, r.total_cost, r.attraction_count])

    agg = compute_aggregate_scores(results)
    ci = bootstrap_aggregate(results, n_boot=1000)
    agg["model"] = label
    agg["n_tasks"] = len(queries)
    agg["n_missing"] = n_missing          # queries with no result row -> scored as failures
    agg["n_error"] = n_err
    for k, v in ci.items():
        agg[k + "_lo"] = v["lo"]
        agg[k + "_hi"] = v["hi"]
    ov, lo, hi = agg.get('weighted_overall'), agg.get('weighted_overall_lo'), agg.get('weighted_overall_hi')
    ci_txt = f" [{lo:.3f},{hi:.3f}]" if isinstance(lo, float) and isinstance(hi, float) else ""
    if agg.get("missing_categories"):
        print(f"  [warn] {label}: no headline score — missing {agg['missing_categories']}")
    if n_missing:
        print(f"  [warn] {label}: {n_missing} queries had NO result row; scored as failures")
    _tok = agg.get('avg_total_tokens')
    avg_tok = f"{round(_tok):,}" if isinstance(_tok, (int, float)) else "—"
    print(f"  {label}: valid_rate={_fmt(agg.get('fully_valid_rate'))}  "
          f"avg_tok={avg_tok}  "
          f"avg_sec={_fmt(agg.get('avg_elapsed_sec'))}")
    print(f"  {label}: n={len(queries)}  weighted_overall={_fmt(ov)}{ci_txt}  "
          f"sat={_fmt(agg.get('cat_satisfaction'))} truth={_fmt(agg.get('cat_truthfulness'))} "
          f"reason={_fmt(agg.get('cat_reasoning'))} eff={_fmt(agg.get('cat_efficiency'))}")
    return agg


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser(description="Score TREK run outputs")
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "trek_results"))
    ap.add_argument("--input", help="Score a single JSONL instead of a whole dir")
    ap.add_argument("--meta", default=os.path.join(os.path.dirname(__file__), "trek_queries.csv"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "trek_scores"))
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k-threshold", type=int, default=15)
    args = ap.parse_args()

    queries = load_queries(args.meta)
    print(f"[queries] {len(queries)} from {args.meta}")
    scorer = TravelPlanScorer(get_sandbox_db(),
                              {"beta": args.beta, "alpha": args.alpha, "k_threshold": args.k_threshold})

    os.makedirs(args.out_dir, exist_ok=True)
    files = [args.input] if args.input else sorted(glob.glob(os.path.join(args.results_dir, "trek_*.jsonl")))
    if not files:
        raise SystemExit(f"[fatal] no result files in {args.results_dir} (expected trek_*.jsonl)")

    summaries = []
    for path in files:
        print(f"\n[scoring] {path}")
        summaries.append(score_file(path, queries, scorer, args.out_dir))

    # combined summary across models
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    cols = (["model", "n_tasks", "n_missing", "n_error"] + DIMS + CATS
            + [c + s for c in CATS[:5] for s in ("_lo", "_hi")])
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n[done] summary -> {summary_csv}")


if __name__ == "__main__":
    main()
