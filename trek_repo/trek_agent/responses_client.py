"""
bedrock-mantle *Responses API* client — the only dialect the GPT-5.x family speaks.

`openai.gpt-5.6-sol` (and 5.5 / 5.4) reject `/openai/v1/chat/completions` outright
("does not support the '/v1/chat/completions' API") and are served exclusively at
  https://bedrock-mantle.{region}.api.aws/openai/v1/responses
verified live from this machine: plain calls and function tools both return HTTP 200, and —
unlike Anthropic — OpenAI models are NOT geo-blocked from this egress.

Like MantleClient, this class exposes the SAME `converse(...) -> ConverseResult` interface as
BedrockClient and translates on the wire, so TrekAgent stays completely unchanged:

  Bedrock history block                ->  Responses `input` item
  ------------------------------------     ------------------------------------------
  user  {text}                         ->  {"role":"user","content": text}
  assistant {text}                     ->  {"role":"assistant","content": text}
  assistant {toolUse}                  ->  {"type":"function_call","call_id","name","arguments"}
  user {toolResult}                    ->  {"type":"function_call_output","call_id","output"}

  Responses `output` item              ->  ConverseResult
  ------------------------------------     ------------------------------------------
  {"type":"message", output_text}      ->  .text (+ assistant {text} block)
  {"type":"function_call"}             ->  .tool_uses (+ assistant {toolUse} block)
  {"type":"reasoning"}                 ->  ignored for history; billed in usage

We run with store=False: the benchmark replays full history itself every turn (the lossless
notebook protocol), so no server-side state may leak between turns or queries.
"""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Optional

from .bedrock_client import ConverseResult, clean_tool_name, strip_channel_markers, payload_chars
from .credentials import AWSCreds


RESPONSES_URL = "https://bedrock-mantle.{region}.api.aws/openai/v1/responses"


class ResponsesClient:
    def __init__(self, creds: AWSCreds, max_retries: int = 5, timeout: int = 600):
        # 600s read timeout for the same reason MantleClient uses it: at large max_output_tokens a
        # reasoning model can legitimately generate for minutes, and gemma's 180s timeout cost 5.3%
        # of its queries before we raised it.
        if not creds.bearer_token:
            raise ValueError("the Responses API needs a Bedrock API key (bearer token); point "
                             "--cred-file at the file containing it (e.g. apieky/bedrock_new.txt).")
        self.token = creds.bearer_token
        self.url = RESPONSES_URL.format(region=creds.region_name)
        self._max_retries = max_retries
        self._timeout = timeout

    # ---- public: same signature as BedrockClient.converse ---------------------------------------
    def converse(self, model_id: str, system_text: Optional[str], messages: list,
                 tool_config: Optional[dict], max_tokens: int, temperature: float = 0.0) -> ConverseResult:
        body = {
            "model": model_id,
            "input": _to_responses_input(messages),
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if system_text:
            body["instructions"] = system_text
        if tool_config:
            body["tools"] = _to_responses_tools(tool_config)
        # temperature deliberately omitted: reasoning-tier GPT-5.x rejects it, and the retry ladder
        # would only waste an attempt discovering that on every single call.

        data = self._post_with_retry(body)
        out = _parse(data)
        out.request_chars = payload_chars(system_text, messages, tool_config)
        return out

    # ---- internals ------------------------------------------------------------------------------
    def _post_with_retry(self, body: dict) -> dict:
        last = None
        for attempt in range(self._max_retries):
            req = urllib.request.Request(
                self.url, data=json.dumps(body).encode("utf-8"), method="POST",
                headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:400]
                last = f"HTTP {e.code}: {detail}"
                low = detail.lower()
                if e.code in (400, 404) and "does not exist" in low:
                    raise RuntimeError(f"responses model not available: {last}")
                # Bedrock's content-safety filter false-positives on ordinary itineraries and says
                # "please retry" itself — same behaviour as on the chat/completions path, where
                # treating it as fatal cost grok 5.5% of its first queries.
                if e.code == 400 and ("content safety" in low or "please retry" in low):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise RuntimeError(f"responses request failed: {last}")
            except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as e:
                # socket.timeout listed explicitly: on Python 3.9 it is neither a TimeoutError nor a
                # URLError subclass, and omitting it let one slow read kill a query with zero
                # retries on the mantle path.
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"responses converse failed after {self._max_retries} attempts: {last}")


# ---- Bedrock <-> Responses translation -----------------------------------------------------------
def _to_responses_tools(tool_config: dict) -> list:
    """Bedrock toolSpec -> Responses function tool (FLATTENED — no nested {"function": ...})."""
    out = []
    for t in tool_config.get("tools", []):
        spec = t.get("toolSpec", {})
        out.append({
            "type": "function",
            "name": spec.get("name"),
            "description": spec.get("description", ""),
            "parameters": (spec.get("inputSchema") or {}).get("json",
                                                              {"type": "object", "properties": {}}),
        })
    return out


def _to_responses_input(messages: list) -> list:
    """Translate Bedrock content-block history into a stateless Responses `input` item list."""
    items = []
    for m in messages:
        role = m.get("role")
        blocks = m.get("content") or []
        if role == "assistant":
            texts = []
            for b in blocks:
                if "text" in b:
                    texts.append(str(b["text"]))
                elif "toolUse" in b:
                    tu = b["toolUse"]
                    items.append({
                        "type": "function_call",
                        "call_id": tu.get("toolUseId"),
                        "name": tu.get("name"),
                        "arguments": json.dumps(tu.get("input", {}) or {}, ensure_ascii=False),
                    })
            if texts:
                items.append({"role": "assistant", "content": "\n".join(texts)})
            continue
        # user turn: toolResult blocks -> function_call_output; plain text -> user message
        texts = []
        for b in blocks:
            if "toolResult" in b:
                tr = b["toolResult"]
                parts = []
                for c in tr.get("content", []):
                    if "json" in c:
                        parts.append(json.dumps(c["json"], ensure_ascii=False))
                    elif "text" in c:
                        parts.append(str(c["text"]))
                items.append({
                    "type": "function_call_output",
                    "call_id": tr.get("toolUseId"),
                    "output": "\n".join(parts) or "{}",
                })
            elif "text" in b:
                texts.append(str(b["text"]))
        if texts:
            items.append({"role": "user", "content": "\n".join(texts)})
    return items


def _parse(data: dict) -> ConverseResult:
    status = data.get("status", "completed")
    text_parts, tool_uses, assistant_content = [], [], []
    for o in data.get("output", []) or []:
        typ = o.get("type")
        if typ == "message":
            for c in o.get("content", []) or []:
                if c.get("type") == "output_text" and c.get("text"):
                    text_parts.append(c["text"])
        elif typ == "function_call":
            nm = clean_tool_name(o.get("name"))
            try:
                args = json.loads(o.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            args = strip_channel_markers(args)
            # call_id (not the item id) is what a later function_call_output must reference.
            tool_uses.append({"toolUseId": o.get("call_id"), "name": nm, "input": args})
            assistant_content.append({"toolUse": {"toolUseId": o.get("call_id"),
                                                  "name": nm, "input": args}})
        # "reasoning" items carry no replayable content (encrypted); tokens show up in usage.

    text = "".join(text_parts)
    if text:
        assistant_content.insert(0, {"text": text})

    u = data.get("usage") or {}
    usage = {
        "prompt_tokens": u.get("input_tokens", 0),
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": u.get("total_tokens",
                              (u.get("input_tokens", 0) or 0) + (u.get("output_tokens", 0) or 0)),
    }
    incomplete = (status == "incomplete"
                  and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens")
    return ConverseResult(
        text=text,
        tool_uses=tool_uses,
        stop_reason=("tool_use" if tool_uses else
                     "max_tokens" if incomplete else "end_turn"),
        usage=usage,
        assistant_content=assistant_content or [{"text": "(no content)"}],
        truncated=incomplete,
    )
