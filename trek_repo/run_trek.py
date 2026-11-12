#!/usr/bin/env python3
"""
TREK full-run driver — all AWS Bedrock, native tool use, multi-model, resumable, max concurrency.

Replaces run_benchmark_full.py (stale queries + hardcoded/leaked keys + single model + no server).
Credentials are loaded at runtime by trek_agent.credentials (from the repo-root `apieky` file or the
AWS default chain) and never printed or written into outputs. The exact Bedrock model IDs are supplied
via --models / --models-file.

Examples:
  # smoke-test the whole chain on 2 queries with one model
  python run_trek.py --models moonshotai.kimi-k2.5 --limit 2

  # full run over several models at high concurrency
  python run_trek.py --models-file trek_models.json --concurrency 16

Before running, start the Travel API (canonical app on the port you pass as --api-url):
  TREK_API_PORT=5001 python api/app.py        # then --api-url http://localhost:5001
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_queries                       # noqa: E402
from trek_agent.credentials import load_credentials        # noqa: E402
from trek_agent.bedrock_client import BedrockClient        # noqa: E402
from trek_agent.mantle_client import MantleClient          # noqa: E402
from trek_agent.agent import TrekAgent, RunConfig          # noqa: E402
from trek_agent import model_config as mc                  # noqa: E402


def make_client(spec, creds):
    """bedrock-runtime (Converse), bedrock-mantle (chat/completions), or the mantle Responses API
    (GPT-5.x — the only dialect that family accepts), per model."""
    ep = spec.resolved_endpoint()
    if ep == "responses":
        from trek_agent.responses_client import ResponsesClient
        return ResponsesClient(creds)
    if ep == "mantle":
        return MantleClient(creds)
    return BedrockClient(creds)


def parse_models(args) -> list:
    specs = []
    if args.models_file:
        with open(args.models_file) as f:
            for m in json.load(f):
                specs.append(mc.ModelSpec(model_id=m["model_id"], label=m.get("label"),
                                          max_tokens=m.get("max_tokens"),
                                          endpoint=m.get("endpoint")))
    if args.models:
        # Accept either an exact Bedrock model ID or a label from trek_models.json. Passing a label
        # used to build a ModelSpec with model_id=<label>, which no endpoint recognises: the run then
        # completed every query with an error record and scored 0 instead of failing fast.
        catalogue = {}
        default_file = os.path.join(os.path.dirname(__file__), "trek_models.json")
        if os.path.exists(default_file):
            with open(default_file) as f:
                for m in json.load(f):
                    catalogue[m.get("label", "")] = m
        for mid in args.models.split(","):
            mid = mid.strip()
            if not mid:
                continue
            m = catalogue.get(mid)
            if m:
                specs.append(mc.ModelSpec(model_id=m["model_id"], label=m.get("label"),
                                          max_tokens=m.get("max_tokens"), endpoint=m.get("endpoint")))
            else:
                specs.append(mc.ModelSpec(model_id=mid))
    if not specs:
        specs = list(mc.MODELS)
    return specs


def preflight(specs: list, creds) -> list:
    """One trivial call per model before the real run.

    An 800-query x 11-model run is hours long; a model that is mistyped, retired, or geo-blocked for
    this account otherwise burns that whole budget producing error records that score 0. This costs
    ~11 tiny requests and turns those failures into an abort at second zero.

    Also records each model's prompt_tokens for an IDENTICAL payload, which is the calibration
    evidence that cross-model token counts are comparable (pilot: max/min = 1.09x).
    """
    probe_sys = "You are a travel planning assistant operating in a sandbox."
    probe_msgs = [{"role": "user", "content": [{"text": "Reply with the single word: ready"}]}]
    ok, bad = [], []
    print("[preflight] probing each model ...")
    for s in specs:
        try:
            r = make_client(s, creds).converse(s.model_id, probe_sys, probe_msgs, None, 16, 0.0)
            ok.append(s)
            print(f"    OK   {s.resolved_label():18} [{s.resolved_endpoint()}] "
                  f"prompt_tokens={r.usage.get('prompt_tokens', 0)}")
        except Exception as e:
            bad.append((s, f"{type(e).__name__}: {e}"))
            print(f"    FAIL {s.resolved_label():18} [{s.resolved_endpoint()}] {str(e)[:120]}")
    if bad:
        raise SystemExit(f"[fatal] {len(bad)} model(s) unreachable; fix or drop them before the run "
                         f"(or pass --no-preflight to run anyway).")
    return ok


def health_check(base_url: str) -> None:
    """Fail fast if the Travel API / the /travel_time route isn't actually serving."""
    url = base_url.rstrip("/") + "/travel_time"
    params = {"from_latitude": 0, "from_longitude": 0, "to_latitude": 0, "to_longitude": 0}
    try:
        r = requests.get(url, params=params, timeout=10)
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"[fatal] Travel API not reachable at {base_url}: {e}\n"
                         f"        Start it first, e.g. TREK_API_PORT=5001 python api/app.py")
    if r.status_code != 200 or "min_travel_minutes" not in r.text:
        raise SystemExit(f"[fatal] /travel_time health check failed (HTTP {r.status_code}): {r.text[:200]}\n"
                         f"        The route may be unregistered (was the 'route after app.run()' bug).")

    # SEMANTIC path too. /travel_time touches no CSV, no faiss and no credentials, so it stayed green
    # while every amenity/facility/extra_service search 500'd on a missing Bedrock credential — and
    # those are 24.8% of all searches the agents make, the ones D1 is scored on. The server must be
    # started with credentials in its environment (see serve_api.py / TREK_CRED_FILE).
    try:
        r2 = requests.get(base_url.rstrip("/") + "/hotels2",
                          params={"city": "Paris", "amenity": "pet-friendly", "top_k": 3}, timeout=60)
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"[fatal] semantic search probe failed: {e}")
    if r2.status_code != 200:
        raise SystemExit(
            f"[fatal] semantic search is broken (HTTP {r2.status_code}). The API process is missing "
            f"Bedrock credentials for the Titan embedder.\n"
            f"        Restart it as: TREK_CRED_FILE=<path to apieky file> python serve_api.py\n"
            f"        Body: {r2.text[:200]}")
    print(f"[ok] Travel API healthy at {base_url} (/travel_time + semantic search responding)")


# After this many consecutive errored queries a model is assumed broken (expired credentials, a
# retired model id, a revoked quota). Without a breaker the driver grinds through all 800 at ~31s of
# retry backoff each — ~26 minutes of wall clock per broken model producing nothing.
CIRCUIT_BREAKER_N = 10


def load_done_indices(path: str) -> set:
    """Query indices already SUCCESSFULLY completed. Errored / un-submitted rows are NOT counted as
    done, so a resume retries them (score_trek dedups by query_index, keeping the latest)."""
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("error") or rec.get("submitted") is False:
                    continue
                if "query_index" in rec:
                    done.add(rec["query_index"])
    return done


def probe_tool_support(client, spec) -> tuple:
    """One tiny Converse call to check the model accepts toolConfig. Returns (ok, reason)."""
    from trek_agent.tools import to_bedrock_toolconfig
    from trek_agent.bedrock_client import user_text_message
    try:
        client.converse(spec.model_id, "Test.", [user_text_message("Reply with the word ok.")],
                        to_bedrock_toolconfig(), 64, 0.0)
        return True, None
    except Exception as e:
        msg = str(e).lower()
        if "tool" in msg and ("support" in msg or "not support" in msg or "toolconfig" in msg):
            return False, str(e)
        return True, None  # transient/other error — let the real run deal with it


def run_one_model(spec: mc.ModelSpec, queries, base_url: str, creds, concurrency: int,
                  out_dir: str, run_cfg: RunConfig) -> None:
    # Re-probe the API before EVERY model, not just once at startup: the run is hours long and a
    # single-process Flask server can die (OOM, laptop sleep, a stray Ctrl-C) in the middle of it.
    health_check(base_url)
    client = make_client(spec, creds)   # clients are thread-safe; shared across workers
    if not mc.supports_tools(spec.model_id):
        print(f"\n=== {spec.model_id} ===\n    [skip] family known to lack tool use; "
              f"this tool-native agent can't benchmark it.")
        return
    ok, reason = probe_tool_support(client, spec)
    if not ok:
        print(f"\n=== {spec.model_id} ===\n    [skip] model rejected tools: {reason[:160]}")
        return
    out_path = os.path.join(out_dir, f"trek_{spec.resolved_label()}.jsonl")
    done = load_done_indices(out_path)
    todo = [(i, q) for i, q in queries if i not in done]
    print(f"\n=== {spec.model_id}  [{spec.resolved_endpoint()}]  "
          f"(max_tokens={spec.resolved_max_tokens()}) ===")
    print(f"    output: {out_path}  |  {len(done)} done, {len(todo)} to run, concurrency={concurrency}")

    write_lock = threading.Lock()
    fh = open(out_path, "a", encoding="utf-8")
    completed = [0]

    consecutive_errors = [0]

    def work(item):
        idx, qmeta = item
        if consecutive_errors[0] >= CIRCUIT_BREAKER_N:
            return                                    # model is down; stop burning the backoff ladder
        agent = TrekAgent(client, base_url, spec, run_cfg)
        t0 = time.monotonic()
        try:
            res = agent.run(qmeta.query)
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}", "plan": {}, "is_feasible": False,
                   "tool_call_count": 0, "submitted": False}

        # An API outage does NOT raise: ToolDispatcher turns a connection error or an HTTP>=400 into
        # a normal-looking {"type": "api_error"} tool result, so the agent plans on empty inventory
        # and usually still submits. Without this the row lands with error=None + submitted=True,
        # load_done_indices counts it DONE, and a relaunch never retries it — an outage at hour 3
        # would silently turn the rest of the run into unusable rows that look like model failures.
        # Key on the api_error marker, NOT on an empty notebook: a correct refusal of an infeasible
        # task legitimately has an empty notebook (verified on 6 pilot rows: searches returned [] and
        # the model correctly refused), so the notebook heuristic would re-run all 267 of those
        # forever.
        # A payload that still carries channel markers after stripping is unusable — extract_plan_data
        # would read 0 entities and the row would look like a genuine planning failure forever.
        from trek_agent.bedrock_client import payload_has_markers
        if not res.get("error") and payload_has_markers(res.get("plan") or {}):
            res["error"] = "channel_markers_in_plan: unparseable submit_plan payload"

        if not res.get("error") and any("api_down" in str(s.get("result_preview", ""))
                                        for s in (res.get("trajectory") or [])):
            res["error"] = "api_unavailable: a tool call could not reach the Travel API"

        if res.get("error"):
            consecutive_errors[0] += 1
        else:
            consecutive_errors[0] = 0
        res["query_index"] = idx
        res["elapsed_sec"] = round(time.monotonic() - t0, 2)
        with write_lock:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            completed[0] += 1
            if completed[0] % 20 == 0 or completed[0] == len(todo):
                print(f"    [{spec.resolved_label()}] {completed[0]}/{len(todo)}")

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(work, it) for it in todo]
            for _ in as_completed(futures):
                pass
    finally:
        fh.close()
    if consecutive_errors[0] >= CIRCUIT_BREAKER_N:
        print(f"    !! CIRCUIT BREAKER: {spec.resolved_label()} failed {consecutive_errors[0]} "
              f"queries in a row and was abandoned. Re-run this model after fixing the cause; "
              f"errored rows are retried automatically on resume.")
    print(f"    done: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="TREK all-Bedrock benchmark driver")
    ap.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "trek_queries.csv"))
    ap.add_argument("--models", default="", help="Comma-separated Bedrock model IDs")
    ap.add_argument("--models-file", default="", help="JSON [{model_id,label?,max_tokens?}]")
    ap.add_argument("--api-url", default=os.environ.get("TREK_API_URL", "http://localhost:5001"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, help="Only the first N queries (after --start)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "trek_results"))
    ap.add_argument("--cred-file", default=os.environ.get("TREK_CRED_FILE"))
    ap.add_argument("--region", default=None)
    ap.add_argument("--max-tool-calls", type=int, default=mc.MAX_TOOL_CALLS)
    ap.add_argument("--no-preflight", action="store_true",
                    help="Skip the one-call-per-model reachability probe (not recommended)")
    args = ap.parse_args()

    specs = parse_models(args)
    if not specs:
        raise SystemExit("[fatal] No models. Pass --models '<id1>,<id2>' or --models-file trek_models.json "
                         "(the exact Bedrock model IDs).")

    creds = load_credentials(path=args.cred_file, region=args.region)
    print(f"[creds] {creds.redacted()}")   # never prints the secret

    queries = list(enumerate(load_queries(args.queries)))
    if args.start or args.limit:
        end = args.start + args.limit if args.limit else len(queries)
        queries = queries[args.start:end]
    print(f"[queries] {len(queries)} from {args.queries}")

    os.makedirs(args.out_dir, exist_ok=True)
    health_check(args.api_url)

    run_cfg = RunConfig(max_tool_calls=args.max_tool_calls)
    print(f"[models] {[s.model_id for s in specs]}")
    if not args.no_preflight:
        specs = preflight(specs, creds)
    for spec in specs:
        run_one_model(spec, queries, args.api_url, creds, args.concurrency, args.out_dir, run_cfg)
    print("\n[all done]")


if __name__ == "__main__":
    main()
