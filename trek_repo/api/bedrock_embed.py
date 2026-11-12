"""Amazon Bedrock (Titan Text Embeddings V2) embedding backend for TravelBench.

Replaces the local SentenceTransformer("Qwen/Qwen3-Embedding-0.6B"). Titan v2 is 1024-dim and
normalised, so the FAISS index code (which reads the dimension off the array) needs no change.

Chosen over Cohere because Cohere embed-v3 is not subscribable on this account (Bedrock
Marketplace AccessDenied). On this KB's persona-retrieval task Titan v2 scored precision@3 = 1.000
including the 23 variant-vocab cities where exact string match drops to 0.892 -- it matches
"Pet Owners" to a query for "with pets", which is the whole reason for using embeddings here.

Credentials come from the standard AWS chain (env: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or
~/.aws/credentials). No secret is stored in code. Region defaults to us-east-1, override with
AWS_REGION / BEDROCK_REGION.

The .encode() method is a drop-in for SentenceTransformer.encode: pass a list of strings, get an
(n, 1024) float32 array back — the shape core_api's faiss calls already expect.
"""
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import boto3
from botocore.config import Config

MODEL_ID = "amazon.titan-embed-text-v2:0"
DIM = 1024
_REGION = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION") or "us-east-1"

# One client, shared across threads (botocore clients are thread-safe for invoke_model). Adaptive
# retry backs off on Bedrock throttling instead of failing the batch.
_local = threading.local()


def _client():
    c = getattr(_local, "client", None)
    if c is None:
        c = boto3.client(
            "bedrock-runtime",
            region_name=_REGION,
            config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
        )
        _local.client = c
    return c


_EMB_CACHE = {}
_EMB_LOCK = threading.Lock()


def _embed_one(text: str) -> np.ndarray:
    """Embed one string, memoised process-wide.

    Semantic search (amenity / facility / extra_service) is 24.8% of every search the agents make,
    and each one was a live Bedrock round-trip. Across 308 real runs only 83 DISTINCT query strings
    ever appeared ("accessible" x41, "pet-friendly" x38, ...), so caching removes essentially the
    whole live dependency: the full 14x800 sweep would otherwise make ~10^4 identical calls, each a
    chance for a throttle or a credential blip to 500 the request and poison the row.
    """
    key = str(text).strip() or "none"
    hit = _EMB_CACHE.get(key)
    if hit is not None:
        return hit
    body = json.dumps({"inputText": key, "dimensions": DIM, "normalize": True})
    resp = _client().invoke_model(modelId=MODEL_ID, body=body)
    v = json.loads(resp["body"].read())["embedding"]
    arr = np.asarray(v, dtype=np.float32)
    with _EMB_LOCK:
        _EMB_CACHE[key] = arr
    return arr


def embed_many(texts, workers: int = 16) -> np.ndarray:
    """Embed a list of strings, deduplicating identical strings so each is called once.

    Returns (len(texts), 1024) float32 aligned to the input order.
    """
    texts = [str(t) for t in texts]
    uniq = list(dict.fromkeys(texts))
    vecs = {}
    if uniq:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for t, v in zip(uniq, ex.map(_embed_one, uniq)):
                vecs[t] = v
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return np.vstack([vecs[t] for t in texts]).astype(np.float32)


class BedrockEncoder:
    """Drop-in for the subset of SentenceTransformer used by core_api: .encode(list[str])."""

    def encode(self, texts, **_):
        if isinstance(texts, str):
            texts = [texts]
        return embed_many(list(texts))


# Module-level singleton so `from bedrock_embed import embedding_model` mirrors the old import.
embedding_model = BedrockEncoder()
