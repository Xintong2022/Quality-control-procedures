"""
The TREK agent loop: a Bedrock-native function-calling ReAct with a lossless notebook and an explicit
final-planning step.

Accounting (matches the scorer's D5): search_* and submit_plan are BILLABLE (count toward
tool_call_count); compute_travel_time and write_note are FREE. The billable count returned here is the
one to feed scoring.py.
"""
from dataclasses import dataclass
from typing import Optional

from .bedrock_client import (BedrockClient, ConverseResult, user_text_message,
                             assistant_message, tool_result_message)
from .tools import (TOOLS, to_bedrock_toolconfig, ToolDispatcher,
                    AUX_TOOLS, SEARCH_TOOLS, TERMINAL_TOOLS)
from .notebook import Notebook
from .prompts import (SYSTEM_PROMPT, build_user_prompt, FINAL_PLANNING_PROMPT,
                      NO_TOOL_NUDGE, BUDGET_EXHAUSTED_NUDGE)
from . import model_config as mc


@dataclass
class RunConfig:
    max_tool_calls: int = mc.MAX_TOOL_CALLS
    max_aux_calls: int = mc.MAX_AUX_CALLS
    max_total_turns: int = mc.MAX_TOTAL_TURNS
    temperature: float = mc.TEMPERATURE
    max_no_tool_nudges: int = 2


class TrekAgent:
    def __init__(self, client: BedrockClient, base_url: str, model_spec: mc.ModelSpec,
                 config: Optional[RunConfig] = None):
        self.client = client
        self.base_url = base_url
        self.model_spec = model_spec
        self.config = config or RunConfig()
        self.tool_config = to_bedrock_toolconfig()

    def run(self, query: str) -> dict:
        notebook = Notebook()
        dispatcher = ToolDispatcher(self.base_url, notebook)
        messages = [user_text_message(build_user_prompt(query))]
        max_tokens = self.model_spec.resolved_max_tokens()

        billable = aux = turns = truncations = no_tool_nudges = 0
        final_planning_injected = False
        submitted = False
        final_plan, is_feasible, refusal_reason = {}, False, ""
        stop_condition = "max_turns"
        trajectory = []
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        request_chars_total = 0   # provider-independent volume measure
        error = None

        while turns < self.config.max_total_turns:
            turns += 1
            try:
                resp: ConverseResult = self.client.converse(
                    self.model_spec.model_id, SYSTEM_PROMPT, messages,
                    self.tool_config, max_tokens, self.config.temperature)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                stop_condition = "error"
                break

            for k in usage_total:
                usage_total[k] += resp.usage.get(k, 0)
            request_chars_total += getattr(resp, "request_chars", 0)
            if resp.truncated:
                truncations += 1

            # echo the model's content blocks back into history (needed to anchor toolResult ids)
            messages.append(assistant_message(resp.assistant_content))

            # -- no tool call: text-only turn -----------------------------------------------------
            if not resp.has_tool_use:
                # DO NOT tell a model that has barely started to "STOP searching". The brief used to
                # fire on the FIRST tool-less turn whatever the progress: measured on 308 real runs,
                # 45 (14.6%) received it while still short of the scorer's own min_calls — 85% of
                # llama4-maverick's runs — and those rows then under-called 89.8% of the time versus
                # 43.6% for the rest. A model that thinks out loud once was being pushed to submit a
                # plan it had not researched. Nudge it back to tool use instead; the brief is for an
                # agent that has genuinely finished (budget exhausted, or repeatedly idle).
                barely_started = billable < max(2, self.config.max_tool_calls // 3)
                if not final_planning_injected and not barely_started:
                    messages.append(user_text_message(
                        FINAL_PLANNING_PROMPT.format(notebook=notebook.render_for_planning())))
                    final_planning_injected = True
                elif no_tool_nudges < self.config.max_no_tool_nudges:
                    no_tool_nudges += 1
                    messages.append(user_text_message(NO_TOOL_NUDGE))
                else:
                    stop_condition = "stalled"
                    break
                continue

            # A turn that DID call a tool clears the strike count: the 3-strike rule is meant to
            # catch an agent stuck emitting prose, not to accumulate across a whole run and kill a
            # model that simply narrates between tool calls.
            no_tool_nudges = 0

            # -- has tool calls: execute each, collecting toolResult blocks -----------------------
            result_blocks = []
            budget_hit_this_turn = False
            for tu in resp.tool_uses:
                name, args, tuid = tu["name"], tu["input"], tu["toolUseId"]
                step = {"turn": turns, "tool": name, "args": args}

                if name in TERMINAL_TOOLS:  # submit_plan
                    final_plan = args.get("plan", {}) or {}
                    is_feasible = bool(args.get("is_feasible", False))
                    refusal_reason = args.get("refusal_reason", "") or ""
                    submitted = True
                    billable += 1          # scorer's D5 min_calls includes submit (min_submit=1)
                    step["result"] = "submitted"
                    step["billable"] = True
                    trajectory.append(step)
                    break

                if name in AUX_TOOLS:
                    if aux >= self.config.max_aux_calls:
                        result = {"error": "aux call cap reached; proceed to submit_plan"}
                    else:
                        aux += 1
                        result = dispatcher.execute(name, args)
                    step["billable"] = False
                elif name in SEARCH_TOOLS:
                    # Bill PER CITY to match the scorer's D5 min_calls (k=cities_count per domain):
                    # a comma-batched city search costs one unit per city, a flight leg costs one.
                    if name == "search_flights":
                        # A round_trip search returns BOTH legs in one response, and the scorer's
                        # min_calls charges k+1 for a round trip. Billing it as 1 unit put 73.6% of
                        # round-trip rows exactly one call under the minimum and docked their D5 for
                        # work they had actually done. 588/800 queries are round trips.
                        n_units = 2 if str(args.get("trip_type", "")).startswith("round") else 1
                    else:
                        cs = [c for c in str(args.get("city", "")).split(",") if c.strip()]
                        n_units = max(1, len(cs))
                    if billable >= self.config.max_tool_calls:
                        result = {"error": "search budget exhausted; compose the plan and submit_plan now"}
                        budget_hit_this_turn = True
                    else:
                        billable += n_units
                        result = dispatcher.execute(name, args)
                        # A search the API could not serve returned nothing, so it must not consume
                        # the agent's efficiency budget: charging it would let a server-side failure
                        # depress D5 and starve the model of the calls it needed. A 4xx (the model
                        # sent a bad request) IS still charged — that is the model's own mistake.
                        if isinstance(result, dict) and result.get("type") == "api_down":
                            billable -= n_units
                    step["billable"] = True
                    step["billed_units"] = 0 if budget_hit_this_turn else (n_units if name != "search_flights" else 1)
                else:
                    result = {"error": f"unknown tool: {name}"}

                step["result_preview"] = _preview(result)
                trajectory.append(step)
                is_err = isinstance(result, dict) and ("error" in result)
                result_blocks.append(tool_result_message(tuid, result, is_error=is_err)["content"][0])

            if submitted:
                stop_condition = "submitted"
                break

            # One user message carrying all toolResults. If it's time to steer (budget hit), append the
            # instruction as a text item INSIDE the last toolResult's content — NOT as a sibling
            # top-level block: several open-weight families reject a turn that mixes toolResult with a
            # standalone text block ("Conversation blocks and tool result blocks cannot be provided in
            # the same turn"). A {text} inside a toolResult's content is a legal ToolResultContentBlock.
            budget_exhausted = billable >= self.config.max_tool_calls
            steer = None
            if (budget_exhausted or budget_hit_this_turn) and not final_planning_injected:
                steer = FINAL_PLANNING_PROMPT.format(notebook=notebook.render_for_planning())
                final_planning_injected = True
            elif budget_exhausted:
                steer = BUDGET_EXHAUSTED_NUDGE
            if steer and result_blocks:
                result_blocks[-1]["toolResult"]["content"].append({"text": steer})
            messages.append({"role": "user", "content": result_blocks})

        return {
            "query": query,
            "model_id": self.model_spec.model_id,
            "model_label": self.model_spec.resolved_label(),
            "is_feasible": is_feasible if submitted else False,
            "refusal_reason": refusal_reason or ("no submission" if not submitted else ""),
            "plan": final_plan or {},
            "tool_call_count": billable,          # <- feed this to scoring.py (D5)
            "aux_call_count": aux,
            "total_turns": turns,
            "submitted": submitted,
            "stop_condition": stop_condition,
            "truncation_events": truncations,
            "trajectory": trajectory,
            "notebook": notebook.to_dict(),
            "token_usage": usage_total,
            "request_chars": request_chars_total,
            "error": error,
        }


def _preview(result) -> str:
    import json
    try:
        s = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(result)
    return s[:300]
