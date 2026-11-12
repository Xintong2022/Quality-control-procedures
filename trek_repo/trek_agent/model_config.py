"""
Per-model run configuration for the all-Bedrock benchmark.

The audit's #1 confound was a single global max_tokens=2048 that silently truncated every reasoning
model (a MAX_TOKENS stop becomes an empty/again-truncated plan → unfair). Here max_tokens is resolved
PER MODEL, generously, so thinking models get room to think AND emit the full day-by-day plan. The
user supplies the exact Bedrock model IDs later; add them to MODELS (or pass --models on the CLI).
Anything not in the table falls back to DEFAULT_MAX_TOKENS, which is already generous.
"""
from dataclasses import dataclass
from typing import Optional
import re


# Generous default: enough for a long multi-city plan even on a non-reasoning model.
DEFAULT_MAX_TOKENS = 8192

# Substring -> max_tokens overrides (first match wins; matched case-insensitively against the model
# id). Reasoning/thinking models get more so thinking tokens don't crowd out the plan. Tune per the
# actual Bedrock model IDs the user provides.
MAX_TOKENS_RULES = [
    # reasoning / thinking families get more room so thinking tokens don't crowd out the plan
    (r"claude.*opus", 16384),
    (r"claude.*sonnet", 12288),
    (r"claude.*(haiku|fable)", 8192),
    (r"gemma-4", 16384),        # Gemma 4: built-in reasoning, 256K ctx (mantle endpoint)
    (r"grok", 16384),           # Grok 4.3: reasoning always-on, 1M ctx (mantle endpoint)
    (r"gpt-oss", 16384),
    (r"nova-premier|nova\.premier", 12288),
    (r"nova-(pro|lite|micro|2)|nova\.(pro|lite|micro)", 8192),
    (r"llama.*(4|3-3|3\.3|maverick|scout)", 8192),
    (r"mistral.*large", 8192),
]

# Bedrock model ids EMPIRICALLY confirmed (probe_bedrock_models.py, 2026-07-20) to reject Converse
# tool use. run_trek probes every model live and skips any that fail, so this is only a fast pre-warn.
# DeepSeek-R1 is the one confirmed no-tools family; DeepSeek-V3.2 and the newer open models
# (Gemma-3, Qwen3, Kimi-K2, GLM, MiniMax, gpt-oss) all ACCEPT toolConfig in our probe — do NOT
# pre-skip them here (their tool-USE reliability is validated by a small real run, not this list).
NO_TOOL_FAMILIES = (
    "deepseek.r1", "deepseek-r1",        # live probe: ValidationException 'doesn't support tool use'
    "gemma-3",                           # live probe: accepts toolConfig but never emits a tool call
    "writer.palmyra-vision", "voxtral",  # vision/audio-only variants
)
# NOTE: NVIDIA Nemotron WAS wrongly listed here — nvidia.nemotron-super-3-120b emits tool calls
# (verified live 2026-07-20). Do not re-add a family without a live probe.


def supports_tools(model_id: str) -> bool:
    """Best-effort: False only for ids confirmed to reject Converse tool use. run_trek probes live."""
    mid = model_id.lower()
    return not any(fam in mid for fam in NO_TOOL_FAMILIES)

# Temperature: 0 for determinism/reproducibility. Some models (reasoning) ignore/forbid it; the
# Bedrock client drops it when a model rejects it.
TEMPERATURE = 0.0

# Max BILLABLE tool calls (searches + submit). compute_travel_time / write_note are free (see agent).
# Kept at 15 to match scoring.py's D5 penalty gradient (150/15), so the run budget and the efficiency
# scale agree; a 3-city full trip needs ~13 (3 legs + 3 hotels + 3 attractions + 3 cars + submit).
MAX_TOOL_CALLS = 15

# Hard ceilings so a stuck agent can't loop forever (aux calls + total model turns).
MAX_AUX_CALLS = 40
MAX_TOTAL_TURNS = 60


# Model-id substrings that live ONLY on the OpenAI-compatible `bedrock-mantle` endpoint
# (https://bedrock-mantle.{region}.api.aws/openai/v1) rather than bedrock-runtime Converse.
# Verified from the AWS model cards + live probes: Gemma 4 family and xAI Grok 4.3 are mantle-only
# (their cards show bedrock-runtime = NO, bedrock-mantle = YES, Converse = NO).
MANTLE_ONLY = ("gemma-4", "grok")


# The GPT-5.x family is mantle-hosted but speaks ONLY the Responses API — it rejects
# /chat/completions outright ("does not support the '/v1/chat/completions' API", verified live).
RESPONSES_ONLY = ("gpt-5.4", "gpt-5.5", "gpt-5.6")


def infer_endpoint(model_id: str) -> str:
    """'responses' for the GPT-5.x family, 'mantle' for OpenAI-dialect-only models (Gemma 4,
    Grok), else 'runtime' (Converse)."""
    mid = model_id.lower()
    if any(f in mid for f in RESPONSES_ONLY):
        return "responses"
    return "mantle" if any(f in mid for f in MANTLE_ONLY) else "runtime"


@dataclass
class ModelSpec:
    """One Bedrock model to benchmark."""
    model_id: str                      # e.g. "us.amazon.nova-pro-v1:0" or "google.gemma-4-31b"
    label: Optional[str] = None        # short slug for output filenames; derived if omitted
    max_tokens: Optional[int] = None   # overrides the rule-based resolution if set
    endpoint: Optional[str] = None     # "runtime" (Converse) | "mantle" (OpenAI dialect); inferred

    def resolved_endpoint(self) -> str:
        return self.endpoint or infer_endpoint(self.model_id)

    def resolved_label(self) -> str:
        if self.label:
            return self.label
        # slugify the model id -> safe filename component
        slug = re.sub(r"[^A-Za-z0-9]+", "-", self.model_id).strip("-")
        return slug

    def resolved_max_tokens(self) -> int:
        if self.max_tokens:
            return self.max_tokens
        return max_tokens_for(self.model_id)


def max_tokens_for(model_id: str) -> int:
    mid = model_id.lower()
    for pattern, mt in MAX_TOKENS_RULES:
        if re.search(pattern, mid):
            return mt
    return DEFAULT_MAX_TOKENS


# Fill this in (or pass --models on the CLI) once the exact Bedrock model IDs are decided.
# Left empty on purpose so a run can't silently benchmark a placeholder model.
MODELS: list[ModelSpec] = []
