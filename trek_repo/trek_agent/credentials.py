"""
Runtime credential loading for the all-Bedrock run.

We NEVER read/print/serialize the plaintext secret. Credentials are loaded from a file (default:
the repo-root `apieky`) or the standard AWS env/instance-profile chain, and handed straight to boto3.
The audit found live keys hardcoded in run_benchmark_full.py and committed under KEY/ and serialized
into every output JSON — this module exists so the new runner never does any of that.

Supported `apieky` formats (auto-detected):
  * AWS console CSV        -> header contains "Access key ID","Secret access key"
  * dotenv / KEY=VALUE     -> AWS_ACCESS_KEY_ID=..., AWS_SECRET_ACCESS_KEY=..., AWS_REGION=...
  * JSON                   -> {"aws_access_key_id": ..., "aws_secret_access_key": ..., "region": ...}
  * two bare lines         -> line1 = access key id (AKIA...), line2 = secret
If no file is found or it lacks keys, we fall back to boto3's default chain (env vars, ~/.aws, IAM
role), so running on an AWS instance profile "just works" with no file at all.
"""
import csv
import io
import json
import os
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_REGION = "us-east-1"   # every selected model was confirmed invocable here (2026-07-20 probe)


@dataclass
class AWSCreds:
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    bearer_token: Optional[str] = None      # Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)
    region_name: str = DEFAULT_REGION
    source: str = "default-chain"

    @property
    def explicit(self) -> bool:
        """True when we hold an explicit key pair (else boto3 uses its own resolution chain)."""
        return bool(self.aws_access_key_id and self.aws_secret_access_key)

    @property
    def is_bearer(self) -> bool:
        return bool(self.bearer_token)

    def boto3_kwargs(self) -> dict:
        """kwargs for boto3.client(); empty (beyond region) means 'use the default chain'.
        A bearer token is NOT passed here — it is applied via the AWS_BEARER_TOKEN_BEDROCK env var
        (see apply_bearer_env), which boto3's Bedrock clients read automatically."""
        kw = {"region_name": self.region_name}
        if self.explicit:
            kw["aws_access_key_id"] = self.aws_access_key_id
            kw["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                kw["aws_session_token"] = self.aws_session_token
        return kw

    def apply_bearer_env(self) -> None:
        """Export the Bedrock API key so boto3's bedrock/bedrock-runtime clients pick it up. Called
        by the client/lister before constructing a boto3 client. No-op for key-pair creds."""
        if self.bearer_token:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = self.bearer_token

    def redacted(self) -> dict:
        """Safe-to-log view — no secret/token value; the id is masked to its last 4 chars."""
        akid = self.aws_access_key_id
        masked = ("…" + akid[-4:]) if akid else None
        return {"aws_access_key_id": masked, "region_name": self.region_name,
                "source": self.source, "auth": ("bearer" if self.is_bearer else
                                                 "key-pair" if self.explicit else "default-chain")}


def _find_default_file() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))          # .../KDD_travelbench/trek_agent
    root = os.path.dirname(here)                               # .../KDD_travelbench
    parent = os.path.dirname(root)                             # project root
    for cand in (os.path.join(parent, "apieky"), os.path.join(root, "apieky"),
                 os.path.join(parent, "apikey"), os.path.join(root, "apikey")):
        if os.path.isfile(cand):
            return cand
    return None


def _looks_like_bearer(line: str) -> bool:
    """A Bedrock API key is one opaque base64-ish token (no '=' KEY=VALUE structure, not an AKIA id).
    We treat a lone long token line as a bearer token."""
    import re as _re
    t = line.strip()
    if not t or "\n" in t:
        return False
    if t.startswith(("AKIA", "ASIA")):
        return False
    # base64 / base64url alphabet, reasonably long
    return len(t) >= 60 and bool(_re.fullmatch(r"[A-Za-z0-9+/_=-]+", t))


def _parse(text: str) -> dict:
    """Return {access_key_id, secret_access_key, session_token?, region?, bearer_token?}."""
    out = {}
    stripped = text.lstrip("﻿").strip()

    # JSON
    if stripped.startswith("{"):
        try:
            j = json.loads(stripped)
            low = {str(k).lower().replace("aws_", ""): v for k, v in j.items()}
            out["access_key_id"] = low.get("access_key_id") or low.get("accesskeyid")
            out["secret_access_key"] = low.get("secret_access_key") or low.get("secretaccesskey")
            out["session_token"] = low.get("session_token")
            out["region"] = low.get("region") or low.get("region_name")
            return {k: v for k, v in out.items() if v}
        except json.JSONDecodeError:
            pass

    # AWS console CSV (header row with "Access key ID"). Strip any BOM off the header cells.
    first = stripped.splitlines()[0] if stripped.splitlines() else ""
    if "access key id" in first.lower():
        reader = csv.DictReader(io.StringIO(stripped))
        for row in reader:
            low = {str(k).replace("﻿", "").strip().lower():
                   (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            out["access_key_id"] = low.get("access key id")
            out["secret_access_key"] = low.get("secret access key")
            out["region"] = low.get("region") or low.get("region_name")
            break
        return {k: v for k, v in out.items() if v}

    # dotenv / KEY=VALUE
    if "=" in stripped:
        env = {}
        for line in stripped.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip().upper()] = v.strip().strip('"').strip("'")
        out["access_key_id"] = env.get("AWS_ACCESS_KEY_ID") or env.get("ACCESS_KEY_ID")
        out["secret_access_key"] = env.get("AWS_SECRET_ACCESS_KEY") or env.get("SECRET_ACCESS_KEY")
        out["session_token"] = env.get("AWS_SESSION_TOKEN")
        out["region"] = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
        if out.get("access_key_id"):
            return {k: v for k, v in out.items() if v}

    # two bare lines: id then secret
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    if len(lines) >= 2 and lines[0].startswith(("AKIA", "ASIA")):
        return {"access_key_id": lines[0], "secret_access_key": lines[1]}

    # a single opaque token -> Bedrock API key (bearer)
    if len(lines) == 1 and _looks_like_bearer(lines[0]):
        return {"bearer_token": lines[0]}

    return {}


def load_credentials(path: Optional[str] = None, region: Optional[str] = None) -> AWSCreds:
    """Load Bedrock credentials. `path` overrides the default apieky location; env wins if set.

    Precedence: explicit AWS_* env vars -> the apieky file -> boto3 default chain.
    """
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION

    # 1) explicit env vars
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return AWSCreds(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=region, source="env",
        )
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return AWSCreds(bearer_token=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
                        region_name=region, source="env-bearer")

    # 2) apieky file (key pair, or a Bedrock API bearer token)
    fpath = path or os.environ.get("TREK_CRED_FILE") or _find_default_file()
    if fpath and os.path.isfile(fpath):
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            parsed = _parse(f.read())
        if parsed.get("access_key_id") and parsed.get("secret_access_key"):
            return AWSCreds(
                aws_access_key_id=parsed["access_key_id"],
                aws_secret_access_key=parsed["secret_access_key"],
                aws_session_token=parsed.get("session_token"),
                region_name=parsed.get("region") or region,
                source=os.path.basename(fpath),
            )
        if parsed.get("bearer_token"):
            return AWSCreds(bearer_token=parsed["bearer_token"],
                            region_name=parsed.get("region") or region,
                            source=os.path.basename(fpath))

    # 3) boto3 default chain (env/instance profile/~/.aws)
    return AWSCreds(region_name=region, source="default-chain")
