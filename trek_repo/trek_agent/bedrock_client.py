"""
Thin wrapper over the AWS Bedrock **Converse** API with native tool use.

One code path serves every Bedrock-hosted model (Claude, Nova, Llama, Mistral, DeepSeek, …): tools go
in `toolConfig`, tool calls come back as `toolUse` content blocks, and results go back as `toolResult`
blocks. A MAX_TOKENS stop is surfaced explicitly (never swallowed into empty text — the exact bug the
audit flagged for Gemini). Credentials come from trek_agent.credentials and are never logged.
"""
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .credentials import AWSCreds

try:
    import boto3
    from botocore.config import Config as _BotoConfig
    from botocore.exceptions import ClientError, BotoCoreError
    HAS_BOTO3 = True
except ImportError:  # allows importing/testing the rest of the package without boto3 installed
    HAS_BOTO3 = False
    ClientError = BotoCoreError = Exception


def payload_chars(system_text, messages, tool_config=None) -> int:
    """Provider-INDEPENDENT size of what we sent this turn, in characters.

    An exactly recomputable trace of request volume that needs no tokenizer choice, so a reader can
    verify from the released artifacts whether a model's token bill reflects longer contexts or a
    different tokenizer.

    Provider token counts remain the reported cost axis — they are the real bill, and a calibration
    probe on an identical 1,991-char payload put cross-vendor tokenizer divergence at only 1.09x.
    This counter is kept as a reproducibility audit trail, not as a substitute: it is measured on the
    REQUEST only, so it cannot see hidden reasoning output, and it is asymmetric across our two
    clients (Converse echoes reasoning content back into `messages`, the OpenAI dialect discards it).

    `tool_config` must be passed: the tool schema is re-sent uncached on EVERY turn and serialises to
    more characters than the system prompt, so omitting it under-counted every request by a constant
    per-turn amount.
    """
    n = len(system_text or "")
    if tool_config:
        n += len(json.dumps(tool_config, ensure_ascii=False))
    for m in messages or []:
        for b in (m.get("content") or []):
            if isinstance(b, dict):
                n += len(json.dumps(b, ensure_ascii=False))
            else:
                n += len(str(b))
    return n


def clean_tool_name(name) -> str:
    """Strip a model's internal channel markers out of a tool name.

    gpt-oss emits Harmony control tokens inside the name itself — "submit_plan<|channel|>final",
    "search_attractions<|channel|>commentary". Converse validates tool names against a strict
    pattern, so echoing one back in the next turn's history raised a ValidationException that killed
    the whole query: 2 of 10 pilot queries for gpt-oss-20b died this way, and the loss would have
    been ours, not the model's — it had picked the right tool every time.
    """
    s = str(name or "")
    for marker in ("<|", "<", "\n"):
        i = s.find(marker)
        if i > 0:
            s = s[:i]
    return s.strip()


def strip_channel_markers(obj):
    """Recursively remove Harmony channel markers from a tool-call payload, keys included.

    gpt-oss leaks `<|channel|>` fragments; on gemma-4-31b they fuse into the submit_plan ARGUMENT
    KEYS — `{"day1": {"current_city:<|": ">Taiyuan to Guilin<|"}}` — so extract_plan_data finds no
    flights, no hotel, no attractions, and the row scores 0 on every dimension while looking like a
    genuine planning failure. Measured on 3 of 34 gemma rows (8.8%), 0 rows for every other model.
    Cleaning the tool NAME (see clean_tool_name) was not enough; the payload needs it too.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            k2 = _strip_marker_text(k)
            out[k2] = strip_channel_markers(v)
        return out
    if isinstance(obj, list):
        return [strip_channel_markers(v) for v in obj]
    if isinstance(obj, str):
        return _strip_marker_text(obj)
    return obj


def _strip_marker_text(s: str) -> str:
    t = str(s)
    if "<|" not in t and "|>" not in t:
        return t
    t = re.sub(r"<\|[^|]*\|>", "", t)     # a complete <|channel|> token
    t = t.replace("<|", "").replace("|>", "")
    return t.strip(" :>")


def payload_has_markers(obj) -> bool:
    """True if any key or value still carries a marker after stripping — the row is unusable."""
    if isinstance(obj, dict):
        return any(payload_has_markers(k) or payload_has_markers(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(payload_has_markers(v) for v in obj)
    return isinstance(obj, str) and ("<|" in obj or "|>" in obj)


@dataclass
class ConverseResult:
    text: str                         # concatenated text blocks
    tool_uses: list                   # [{"toolUseId","name","input"}]
    stop_reason: str                  # tool_use | end_turn | max_tokens | stop_sequence | content_filtered
    usage: dict                       # {prompt_tokens, completion_tokens, total_tokens}
    assistant_content: list = field(default_factory=list)  # raw content blocks to append to history
    request_chars: int = 0        # provider-independent size of the request we sent
    truncated: bool = False           # stop_reason == max_tokens

    @property
    def has_tool_use(self) -> bool:
        return bool(self.tool_uses)


class BedrockClient:
    def __init__(self, creds: AWSCreds, max_retries: int = 5, read_timeout: int = 120):
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for the Bedrock run. pip install boto3")
        self.creds = creds
        self._max_retries = max_retries
        creds.apply_bearer_env()   # exports AWS_BEARER_TOKEN_BEDROCK if this is a Bedrock API key
        cfg = _BotoConfig(retries={"max_attempts": 3, "mode": "adaptive"},
                          read_timeout=read_timeout, connect_timeout=15)
        self.client = boto3.client("bedrock-runtime", config=cfg, **creds.boto3_kwargs())

    def converse(self, model_id: str, system_text: Optional[str], messages: list,
                 tool_config: Optional[dict], max_tokens: int, temperature: float = 0.0) -> ConverseResult:
        params: dict = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system_text:
            params["system"] = [{"text": system_text}]
        if tool_config:
            params["toolConfig"] = tool_config

        resp = self._call_with_retry(params)
        out = self._parse(resp)
        out.request_chars = payload_chars(system_text, messages, tool_config)
        return out

    # ---- internals -----------------------------------------------------------------------------
    def _call_with_retry(self, params: dict) -> dict:
        last_err = None
        drop_temp = False
        for attempt in range(self._max_retries):
            try:
                p = dict(params)
                if drop_temp:
                    ic = dict(p["inferenceConfig"]); ic.pop("temperature", None); p["inferenceConfig"] = ic
                return self.client.converse(**p)
            except ClientError as e:
                last_err = e
                code = e.response.get("Error", {}).get("Code", "") if hasattr(e, "response") else ""
                msg = str(e)
                # Some models reject temperature (esp. reasoning models). Drop it and retry once.
                if not drop_temp and code == "ValidationException" and \
                        ("temperature" in msg.lower() or "top_p" in msg.lower()):
                    drop_temp = True
                    continue
                # Throttling / transient -> backoff. Other validation errors -> stop early.
                if code in ("ThrottlingException", "ServiceUnavailableException",
                            "ModelTimeoutException", "InternalServerException", "ModelErrorException"):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if code == "ValidationException":
                    raise
                time.sleep(min(2 ** attempt, 30))
            except BotoCoreError as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Bedrock converse failed after {self._max_retries} attempts: {last_err}")

    @staticmethod
    def _parse(resp: dict) -> ConverseResult:
        stop_reason = resp.get("stopReason", "end_turn")
        out_msg = resp.get("output", {}).get("message", {})
        content_blocks = out_msg.get("content", []) or []

        text_parts, tool_uses = [], []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                # Sanitise IN PLACE as well: `content_blocks` is echoed back verbatim as the next
                # turn's assistant message, so a dirty name would be re-sent and rejected there.
                tu["name"] = clean_tool_name(tu.get("name"))
                tu["input"] = strip_channel_markers(tu.get("input", {}) or {})
                tool_uses.append({
                    "toolUseId": tu.get("toolUseId"),
                    "name": tu["name"],
                    "input": tu["input"],
                })

        u = resp.get("usage", {}) or {}
        usage = {
            "prompt_tokens": u.get("inputTokens", 0),
            "completion_tokens": u.get("outputTokens", 0),
            "total_tokens": u.get("totalTokens", u.get("inputTokens", 0) + u.get("outputTokens", 0)),
        }
        return ConverseResult(
            text="".join(text_parts),
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            usage=usage,
            assistant_content=content_blocks,
            # newer Converse also emits 'model_context_window_exceeded'; treat it as truncation too
            truncated=(stop_reason in ("max_tokens", "model_context_window_exceeded")),
        )


# ---- message construction helpers (Converse content-block format) ------------------------------
def user_text_message(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def assistant_message(content_blocks: list) -> dict:
    """Echo the model's own content blocks back as history (required so toolResult can reference the
    toolUseId). Bedrock rejects empty text blocks, so drop any and fall back to a non-empty
    placeholder if the model returned nothing (e.g. a max_tokens truncation with no content)."""
    # Drop empty text blocks (Bedrock rejects them) AND toolUse blocks with no name. Some models
    # emit a nameless toolUse; clean_tool_name() then normalises it to "", and echoing that back
    # fails validation with `toolUse.name, value: 0, valid min length: 1`, killing the query on the
    # next turn — 11 of mistral-large-3's first 480 queries (2.3%). An unnamed tool call is
    # unusable anyway, so it must not enter the history.
    blocks = [b for b in (content_blocks or [])
              if not (isinstance(b, dict) and set(b.keys()) == {"text"} and not str(b["text"]).strip())
              and not (isinstance(b, dict) and "toolUse" in b
                       and not str((b.get("toolUse") or {}).get("name") or "").strip())]
    return {"role": "assistant", "content": blocks or [{"text": "(no content)"}]}


def tool_result_message(tool_use_id: str, result: Any, is_error: bool = False) -> dict:
    """A user-role message carrying one tool result block."""
    # Converse requires toolResult.content[].json to be a JSON OBJECT. Several of our search
    # endpoints return a top-level LIST (e.g. /hotels2 -> [{...}, {...}]), and passing that through
    # raised "The format of the value at ...toolResult.content.0.json is invalid" on the very next
    # turn — which killed EVERY bedrock-runtime model at turn 2 while the mantle models (which
    # serialise to text) were unaffected. Lists are wrapped so the field is always an object.
    if isinstance(result, dict):
        block = {"json": result}
    elif isinstance(result, list):
        block = {"json": {"results": result}}
    else:
        block = {"text": str(result)}
    tr = {"toolUseId": tool_use_id, "content": [block]}
    if is_error:
        tr["status"] = "error"
    else:
        tr["status"] = "success"
    return {"role": "user", "content": [{"toolResult": tr}]}
