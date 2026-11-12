"""
bedrock-mantle client — the OpenAI-compatible Bedrock endpoint.

Some models (notably the Gemma 4 family) are available ONLY on `bedrock-mantle`, which speaks the
OpenAI Chat Completions dialect at https://bedrock-mantle.{region}.api.aws/openai/v1 and authenticates
with a Bedrock API key (bearer token) — not SigV4 Converse.

This class exposes the SAME `converse(...) -> ConverseResult` interface as BedrockClient, translating
Bedrock-shaped messages/tools to OpenAI on the wire and translating the response back. That keeps
TrekAgent completely unchanged: the agent keeps thinking in Bedrock content blocks, and only this
client knows about the OpenAI dialect.

Per AWS docs, mantle does NOT support parallel tool calls, so we send parallel_tool_calls=false and
surface at most one toolUse per turn.
"""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Optional

from .bedrock_client import ConverseResult, clean_tool_name, strip_channel_markers
from .credentials import AWSCreds


MANTLE_URL = "https://bedrock-mantle.{region}.api.aws/openai/v1/chat/completions"


class MantleClient:
    def __init__(self, creds: AWSCreds, max_retries: int = 5, timeout: int = 600):
        # 600s, not 180s. At max_tokens=32768 a mantle model can legitimately generate for minutes:
        # gemma-4-31b hit the 180s read timeout on 26 of its first 492 queries (5.3%), and because
        # each one then burned the full 5-attempt ladder (5 x 180s) it also dragged the model's
        # throughput from 115 to 226 s/query. The timeout was ours, not the model's.
        if not creds.bearer_token:
            raise ValueError("bedrock-mantle requires a Bedrock API key (bearer token). Point "
                             "--cred-file at the file containing it (e.g. apieky/bedrock_new.txt).")
        self.token = creds.bearer_token
        self.url = MANTLE_URL.format(region=creds.region_name)
        self._max_retries = max_retries
        self._timeout = timeout

    # ---- public: same signature as BedrockClient.converse ---------------------------------------
    def converse(self, model_id: str, system_text: Optional[str], messages: list,
                 tool_config: Optional[dict], max_tokens: int, temperature: float = 0.0) -> ConverseResult:
        body = {
            "model": model_id,
            "messages": _to_openai_messages(system_text, messages),
            "max_completion_tokens": max_tokens,
        }
        if tool_config:
            body["tools"] = _to_openai_tools(tool_config)
            body["parallel_tool_calls"] = False   # mantle: one tool call per turn
        if temperature is not None:
            body["temperature"] = temperature

        data = self._post_with_retry(body)
        out = _parse(data)
        from .bedrock_client import payload_chars
        out.request_chars = payload_chars(system_text, messages, tool_config)
        return out

    # ---- internals ------------------------------------------------------------------------------
    def _post_with_retry(self, body: dict) -> dict:
        last = None
        drop_temp = False
        for attempt in range(self._max_retries):
            b = dict(body)
            if drop_temp:
                b.pop("temperature", None)
            req = urllib.request.Request(
                self.url, data=json.dumps(b).encode("utf-8"), method="POST",
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:400]
                last = f"HTTP {e.code}: {detail}"
                low = detail.lower()
                if not drop_temp and "temperature" in low:
                    drop_temp = True
                    continue
                if e.code in (400, 404) and "does not exist" in low:
                    raise RuntimeError(f"mantle model not available: {last}")
                # Bedrock's automated content-safety filter spuriously rejects ordinary travel
                # itineraries and says so itself ("please retry"). Treated as fatal it killed 3 of
                # grok-4.3's first 55 queries (5.5%); across the two mantle models over 800 queries
                # that is ~80 tasks lost to a transient false positive, recorded as if the model had
                # failed to plan.
                if e.code == 400 and ("content safety" in low or "please retry" in low):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if e.code in (429, 500, 502, 503, 504):   # throttle / transient -> back off
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise RuntimeError(f"mantle request failed: {last}")
            except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
                # socket.timeout MUST be listed explicitly. On Python 3.9 (this environment) it is
                # NOT a subclass of TimeoutError and NOT a subclass of URLError, so the old
                # `except (URLError, TimeoutError)` let a read timeout escape the retry loop
                # entirely — one slow response killed the query with zero retries, on the two
                # mantle-only models (grok-4.3, gemma-4-31b). OSError is the common base and covers
                # connection resets too.
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"mantle converse failed after {self._max_retries} attempts: {last}")


# ---- Bedrock <-> OpenAI translation --------------------------------------------------------------
def _to_openai_tools(tool_config: dict) -> list:
    out = []
    for t in tool_config.get("tools", []):
        spec = t.get("toolSpec", {})
        out.append({"type": "function", "function": {
            "name": spec.get("name"),
            "description": spec.get("description", ""),
            "parameters": (spec.get("inputSchema") or {}).get("json", {"type": "object", "properties": {}}),
        }})
    return out


def _to_openai_messages(system_text: Optional[str], messages: list) -> list:
    """Translate Bedrock content-block messages into OpenAI chat messages."""
    out = []
    if system_text:
        out.append({"role": "system", "content": system_text})
    for m in messages:
        role = m.get("role")
        blocks = m.get("content") or []
        if role == "assistant":
            texts, calls = [], []
            for b in blocks:
                if "text" in b:
                    texts.append(b["text"])
                elif "toolUse" in b:
                    tu = b["toolUse"]
                    calls.append({"id": tu.get("toolUseId"), "type": "function", "function": {
                        "name": tu.get("name"), "arguments": json.dumps(tu.get("input", {}), ensure_ascii=False)}})
            msg = {"role": "assistant", "content": ("".join(texts) or None)}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue

        # user turn: may carry toolResult blocks (-> OpenAI 'tool' messages) and/or plain text
        tool_msgs, texts = [], []
        for b in blocks:
            if "toolResult" in b:
                tr = b["toolResult"]
                parts = []
                for c in tr.get("content", []):
                    if "json" in c:
                        parts.append(json.dumps(c["json"], ensure_ascii=False))
                    elif "text" in c:
                        parts.append(str(c["text"]))
                tool_msgs.append({"role": "tool", "tool_call_id": tr.get("toolUseId"),
                                  "content": "\n".join(parts) or "{}"})
            elif "text" in b:
                texts.append(b["text"])
        out.extend(tool_msgs)
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


_FINISH_MAP = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens",
               "content_filter": "content_filtered"}


def _parse(data: dict) -> ConverseResult:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    finish = choice.get("finish_reason", "stop")
    stop_reason = _FINISH_MAP.get(finish, finish)

    text = msg.get("content") or ""
    tool_uses, assistant_content = [], []
    if text:
        assistant_content.append({"text": text})
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {}) or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        nm = clean_tool_name(fn.get("name"))   # see clean_tool_name: gpt-oss leaks channel markers
        args = strip_channel_markers(args)     # ...and into the ARGUMENT keys on gemma-4-31b
        tool_uses.append({"toolUseId": tc.get("id"), "name": nm, "input": args})
        assistant_content.append({"toolUse": {"toolUseId": tc.get("id"), "name": nm,
                                              "input": args}})
    if tool_uses:
        stop_reason = "tool_use"

    u = data.get("usage", {}) or {}
    usage = {"prompt_tokens": u.get("prompt_tokens", 0),
             "completion_tokens": u.get("completion_tokens", 0),
             "total_tokens": u.get("total_tokens", 0)}
    return ConverseResult(text=text, tool_uses=tool_uses, stop_reason=stop_reason, usage=usage,
                          assistant_content=assistant_content, truncated=(finish == "length"))
