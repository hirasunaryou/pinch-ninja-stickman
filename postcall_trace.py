"""
Postcall trace bundle writer.
----------------------------
This utility builds a developer-only trace bundle under
``.dal_logs/postcall/runs/YYYYMMDD/<requestId>/``. The bundle is intentionally
redacted: it never writes raw transcripts, prompts, or model outputs, while
still preserving enough structure to debug how a pipeline behaved end-to-end.

Usage (high level):
- Prepare a *redacted* OpenAI request payload (messages with sensitive content
  already masked) plus response metadata.
- Pass raw prompt text only to compute ``promptHashRaw``; the text is discarded
  immediately after hashing and is never written to disk.
- Provide sanitized trace snapshots: raw → sanitized → final along with per-field
  sanitize reasons.
- Call :func:`PostcallTraceLogger.write_bundle` with a unique request id.

The resulting directory includes:
05_system_prompt.txt, 06_openai_request.json, 21_openai_response_meta.json,
41_trace_raw.json, 42_trace_sanitized.json, 43_trace_final.json,
44_sanitize_reasons.json, and index.json (with new metadata fields).

Developer note:
This module uses only the standard library and keeps extra inline comments so
future contributors can reason about the redaction rules quickly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256_hex(value: str) -> str:
    """Return a SHA-256 hex digest for consistent hash fields."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _today_str() -> str:
    """Date helper for folder naming (UTC keeps logs timezone-agnostic)."""
    return datetime.utcnow().strftime("%Y%m%d")


def _ensure_redacted_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Shallowly redact common sensitive keys in a message list.

    The goal is defensive: even if the caller accidentally passes raw content,
    we mask the obvious places before writing to disk. This keeps the bundle
    safe-by-default while still preserving message order/roles.
    """
    redacted_messages: List[Dict[str, Any]] = []
    for message in messages:
        copy = dict(message)
        # Mask popular content fields; callers can still provide pre-redacted
        # strings like "[REDACTED]" which will be preserved as-is.
        for key in ("content", "text", "transcript", "raw"):
            if key in copy:
                copy[key] = "[REDACTED]"
        redacted_messages.append(copy)
    return redacted_messages


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Small helper to write pretty JSON with UTF-8 encoding."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    """Write plain UTF-8 text to disk."""
    path.write_text(text, encoding="utf-8")


def _safe_git_commit() -> Optional[str]:
    """
    Try to capture the current git commit hash without failing the bundle.

    This runs in ``--quiet`` mode to avoid noisy stderr output when git is
    unavailable (e.g., packaged builds). Returning ``None`` signals "not
    available", which index.json records explicitly.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


@dataclass
class TraceBundlePaths:
    """
    File layout helper for a single request bundle.

    Keeping paths centralized prevents typos and makes the index generation
    straightforward for future schema updates.
    """

    root: Path

    @property
    def system_prompt(self) -> Path:
        return self.root / "05_system_prompt.txt"

    @property
    def openai_request(self) -> Path:
        return self.root / "06_openai_request.json"

    @property
    def openai_response_meta(self) -> Path:
        return self.root / "21_openai_response_meta.json"

    @property
    def trace_raw(self) -> Path:
        return self.root / "41_trace_raw.json"

    @property
    def trace_sanitized(self) -> Path:
        return self.root / "42_trace_sanitized.json"

    @property
    def trace_final(self) -> Path:
        return self.root / "43_trace_final.json"

    @property
    def sanitize_reasons(self) -> Path:
        return self.root / "44_sanitize_reasons.json"

    @property
    def index(self) -> Path:
        return self.root / "index.json"

    def as_index_paths(self) -> Dict[str, str]:
        """
        Convert to relative strings for index.json so consumers can resolve
        files without hardcoding names.
        """
        return {
            "systemPrompt": self.system_prompt.name,
            "openaiRequest": self.openai_request.name,
            "openaiResponseMeta": self.openai_response_meta.name,
            "traceRaw": self.trace_raw.name,
            "traceSanitized": self.trace_sanitized.name,
            "traceFinal": self.trace_final.name,
            "sanitizeReasons": self.sanitize_reasons.name,
        }


class PostcallTraceLogger:
    """
    Orchestrates writing a complete postcall trace bundle.

    Parameters
    ----------
    log_root:
        Base directory for bundles (defaults to ``.dal_logs/postcall/runs``).
    prompt_version / schema_version:
        Manual strings so operators can bump them intentionally.
    app_version:
        Optional application version to store alongside git commit hash.
    """

    def __init__(
        self,
        *,
        log_root: Path | str = Path(".dal_logs/postcall/runs"),
        prompt_version: str = "v1",
        schema_version: str = "2024-07-01",
        app_version: Optional[str] = None,
    ) -> None:
        self.log_root = Path(log_root)
        self.prompt_version = prompt_version
        self.schema_version = schema_version
        self.app_version = app_version

    def write_bundle(
        self,
        *,
        request_id: str,
        raw_prompt_text: str,
        redacted_prompt_messages: List[Dict[str, Any]],
        system_prompt_redacted: str,
        openai_request_redacted: Dict[str, Any],
        openai_response_meta: Dict[str, Any],
        trace_raw: Dict[str, Any],
        trace_sanitized: Dict[str, Any],
        trace_final: Dict[str, Any],
        sanitize_reasons: Dict[str, Any],
        response_format_name: Optional[str] = None,
        git_commit: Optional[str] = None,
    ) -> Path:
        """
        Write all bundle artifacts for a single request.

        The raw prompt text is only used to compute ``promptHashRaw``; it is
        deliberately never written to disk to satisfy the "never store
        unredacted prompt" constraint.
        """
        date_bucket = _today_str()
        bundle_root = self.log_root / date_bucket / request_id
        bundle_root.mkdir(parents=True, exist_ok=True)

        paths = TraceBundlePaths(bundle_root)

        # System prompt stays textual for quick reading; assume caller already
        # scrubbed sensitive tokens because we never want to store secrets here.
        _write_text(paths.system_prompt, system_prompt_redacted)

        # Normalize and defensively redact the OpenAI request payload.
        redacted_messages = _ensure_redacted_messages(redacted_prompt_messages)
        openai_request_payload = {
            "model": openai_request_redacted.get("model"),
            "messages": redacted_messages,
            "response_format": {
                "name": response_format_name or openai_request_redacted.get("response_format", {}).get("name")
            },
        }
        _write_json(paths.openai_request, openai_request_payload)

        # Response metadata is limited to id/model/usage to avoid leaking
        # generated text.
        response_meta_payload = {
            "id": openai_response_meta.get("id"),
            "model": openai_response_meta.get("model"),
            "usage": openai_response_meta.get("usage"),
            "generatedAt": openai_response_meta.get("generatedAt") or datetime.utcnow().isoformat(),
        }
        _write_json(paths.openai_response_meta, response_meta_payload)

        # Trace snapshots: each stage is kept separate so reviewers can
        # visually diff the transformation pipeline.
        _write_json(paths.trace_raw, trace_raw)
        _write_json(paths.trace_sanitized, trace_sanitized)
        _write_json(paths.trace_final, trace_final)
        _write_json(paths.sanitize_reasons, sanitize_reasons)

        # Hashes: raw prompt hash uses the unredacted text; redacted hash uses
        # the JSON snapshot so it can be compared safely.
        prompt_hash_raw = _sha256_hex(raw_prompt_text)
        prompt_hash_redacted = _sha256_hex(json.dumps(redacted_messages, sort_keys=True))

        resolved_git_commit = git_commit or _safe_git_commit()

        index_payload = {
            "requestId": request_id,
            "createdAt": datetime.utcnow().isoformat(),
            "promptHashRaw": prompt_hash_raw,
            "promptHashRedacted": prompt_hash_redacted,
            "promptVersion": self.prompt_version,
            "schemaVersion": self.schema_version,
            "appVersion": self.app_version,
            "gitCommit": resolved_git_commit,
            "files": paths.as_index_paths(),
        }
        _write_json(paths.index, index_payload)

        return bundle_root


if __name__ == "__main__":
    # Quick demo showing how to write a fully redacted bundle with placeholder
    # values. Running this module directly will create a safe sample bundle that
    # contains no user data.
    logger = PostcallTraceLogger(prompt_version="demo-v1", schema_version="2024-07-01")
    logger.write_bundle(
        request_id="demo-request",
        raw_prompt_text="safe demo prompt for hashing only",
        redacted_prompt_messages=[
            {"role": "system", "content": "[REDACTED SYSTEM PROMPT]"},
            {"role": "user", "content": "[REDACTED TRANSCRIPT SNIPPET]"},
        ],
        system_prompt_redacted="You are a helpful assistant (demo).",
        openai_request_redacted={"model": "gpt-4o-mini", "response_format": {"name": "json_object"}},
        openai_response_meta={"id": "response-123", "model": "gpt-4o-mini", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        trace_raw={"input": "placeholder raw trace"},
        trace_sanitized={"input": "placeholder sanitized trace"},
        trace_final={"output": "placeholder final trace"},
        sanitize_reasons={"input": {"reason": "demo_only"}},
        response_format_name="json_object",
    )
