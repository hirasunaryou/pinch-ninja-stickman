"""
Developer log viewer web UI.
----------------------------
This module spins up a tiny HTTP server that renders postcall trace bundles
under ``.dal_logs/postcall/runs`` in a more investigative, tabbed UI. The goal
is to make it easy to answer “why did this happen?” by walking from the input
all the way to the final output and UI patch for a single ``requestId``.

Launch with:
    python dev_log_viewer.py --port 8000

Key endpoints (all served from this single file to stay dependency-free):
- /dev/logs/                      → List view with filters
- /dev/logs/<date>/<requestId>    → Detail view (shares the same HTML shell)
- /dev/logs/api/runs              → JSON inventory of bundles + derived stats
- /dev/logs/api/run/<date>/<id>   → JSON detail payload for the UI tabs

Why plain http.server instead of a heavier framework?
- Keeps requirements lean (no extra pip installs).
- Keeps the footprint small so contributors can understand and extend it
  quickly. Plenty of inline comments call out the parsing heuristics so readers
  know where to tweak behaviors for their own log schema.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import urllib.parse
import zipfile
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Root where postcall_trace.py writes bundles. Kept configurable for tests.
LOG_ROOT = Path(".dal_logs/postcall/runs")
NOTES_PATH = LOG_ROOT.parent / "70_notes.md"
PROBE_PLAN_QUEUE_PATH = LOG_ROOT.parent / "probe_plan_review_queue.jsonl"


def _today_str() -> str:
    """Date helper for naming notes or defaulting date buckets."""
    return datetime.utcnow().strftime("%Y%m%d")


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    """Return JSON content if the file exists, otherwise an empty dict."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Defensive: malformed JSON should not crash the UI; report as empty.
        return {}


def _read_text_if_exists(path: Path) -> str:
    """Return UTF-8 text if present; otherwise an empty string."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _gather_bundle_paths() -> List[Tuple[str, Path]]:
    """
    Find all bundles under LOG_ROOT and return a list of (date_bucket, path).

    The directory layout is expected to be:
    .dal_logs/postcall/runs/YYYYMMDD/<requestId>/...
    """
    bundles: List[Tuple[str, Path]] = []
    if not LOG_ROOT.exists():
        return bundles
    for date_bucket in sorted(LOG_ROOT.iterdir()):
        if not date_bucket.is_dir():
            continue
        for request_dir in sorted(date_bucket.iterdir()):
            if request_dir.is_dir():
                bundles.append((date_bucket.name, request_dir))
    return bundles


def _find_bundle_by_request_id(request_id: str) -> Optional[Tuple[str, Path]]:
    """
    Locate a bundle by request id across all date buckets.

    The search is intentionally linear because bundle counts stay small in the
    dev-only viewer. Keeping it simple avoids extra indexes while still being
    easy to read and extend for future contributors.
    """
    for date_bucket, run_path in _gather_bundle_paths():
        if run_path.name == request_id:
            return date_bucket, run_path
    return None


def _zip_bundle(path: Path) -> bytes:
    """
    Create an in-memory ZIP archive of a bundle directory.

    Using BytesIO keeps the endpoint dependency-free while still letting the UI
    stream a file download response.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(path.parent)
                # Educational comment: arcname preserves the requestId folder so
                # the unzip path stays clear to the reader.
                zf.write(file_path, arcname)
    return buffer.getvalue()


def _dev_bundle_download_enabled() -> bool:
    """
    Gate bundle downloads behind explicit dev flags.

    We mirror the Node-style environment contract so the Python viewer behaves
    like the intended hosting environment.
    """
    return os.environ.get("NODE_ENV") == "development" and os.environ.get("DAL_LOG_UI") == "true"


def _dev_mutation_enabled() -> bool:
    """
    Reuse the same dev-only guard for write endpoints (notes, review queue).

    Keeping a single toggle mirrors the download protection above and makes the
    expected environment contract easy to remember while hacking locally.
    """
    return _dev_bundle_download_enabled()


def _token_count(meta: Dict[str, Any]) -> Optional[int]:
    """
    Calculate total tokens from an OpenAI usage object if present.
    """
    usage = meta.get("usage") if isinstance(meta, dict) else {}
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt + completion
    if isinstance(prompt, int):
        return prompt
    return None


def _compute_missing_count(final_trace: Dict[str, Any], sanitize_report: Dict[str, Any]) -> int:
    """
    Estimate how many extraction fields look incomplete.

    We combine two signals:
    - sanitize_report per-field reasons that indicate missing/invalid evidence.
    - final_trace perField entries that lack a value.
    """
    missing = 0
    per_field_report = sanitize_report.get("perField", []) if isinstance(sanitize_report, dict) else []
    for field in per_field_report:
        reason = field.get("reason")
        matched_by = field.get("matchedBy")
        if reason in {"invalid_turnIndex", "snippet_not_found", "value_not_found"} or matched_by == "none":
            missing += 1

    extraction_trace = final_trace.get("extractionTrace", {}) if isinstance(final_trace, dict) else {}
    for field in extraction_trace.get("perField", []):
        if not field.get("value") and not field.get("requestedSnippet"):
            missing += 1
    return missing


def _extract_tags(request_json: Dict[str, Any], index_json: Dict[str, Any]) -> List[str]:
    """
    Collect tag-like labels from either the request payload or the index.
    """
    tags: List[str] = []
    for source in (request_json, index_json):
        for key in ("tags", "labels", "tag"):
            candidate = source.get(key)
            if isinstance(candidate, list):
                tags.extend([str(item) for item in candidate])
            elif isinstance(candidate, str):
                tags.append(candidate)
    return tags


def _extract_transcript(request_json: Dict[str, Any], trace_final: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a normalized transcript list of {idx, speaker, text}.
    """
    candidates: List[Any] = []
    # Common places the transcript may live.
    for key in ("transcriptTurns", "transcript"):
        if key in request_json:
            candidates.append(request_json[key])
    input_block = request_json.get("input", {})
    if isinstance(input_block, dict):
        for key in ("transcriptTurns", "transcript"):
            if key in input_block:
                candidates.append(input_block[key])
    extraction_trace = trace_final.get("extractionTrace", {}) if isinstance(trace_final, dict) else {}
    if "turns" in extraction_trace:
        candidates.append(extraction_trace["turns"])

    def normalize_turn(turn: Any, idx: int) -> Optional[Dict[str, Any]]:
        if isinstance(turn, dict):
            text = turn.get("text") or turn.get("content") or turn.get("message")
            if text is None:
                return None
            speaker = turn.get("speaker") or turn.get("role") or turn.get("sender") or "unknown"
            return {"idx": idx, "speaker": speaker, "text": str(text)}
        if isinstance(turn, str):
            return {"idx": idx, "speaker": "unknown", "text": turn}
        return None

    for candidate in candidates:
        if isinstance(candidate, list) and candidate:
            transcript: List[Dict[str, Any]] = []
            for idx, turn in enumerate(candidate):
                normalized = normalize_turn(turn, idx)
                if normalized:
                    transcript.append(normalized)
            if transcript:
                return transcript
    return []


def _extract_evidence_map(trace_final: Dict[str, Any], sanitize_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Combine per-field extraction info with sanitize reasons for a single table.
    """
    extraction_trace = trace_final.get("extractionTrace", {}) if isinstance(trace_final, dict) else {}
    per_field = extraction_trace.get("perField", []) if isinstance(extraction_trace, dict) else []
    reasons = sanitize_report.get("perField", []) if isinstance(sanitize_report, dict) else []
    reasons_by_field = {}
    for reason in reasons:
        name = reason.get("field")
        if name not in reasons_by_field:
            reasons_by_field[name] = []
        reasons_by_field[name].append(reason)

    rows: List[Dict[str, Any]] = []
    for entry in per_field:
        field_name = entry.get("field") or entry.get("name") or entry.get("path") or ""
        reason_candidates = reasons_by_field.get(field_name) or []
        reason = reason_candidates[0] if reason_candidates else {}
        rows.append(
            {
                "field": field_name,
                "status": reason.get("reason") or reason.get("matchedBy") or "ok",
                "turnIndex": entry.get("turnIndex"),
                "snippet": entry.get("requestedSnippet") or entry.get("snippet") or "",
                "value": entry.get("value"),
                "matchedBy": reason.get("matchedBy"),
                "turnTextPreview": reason.get("turnTextPreview"),
                "reason": reason.get("reason"),
            }
        )
    return rows


def _extract_assets(request_json: Dict[str, Any], final_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull asset hint input vs linked assets output for side-by-side comparison.
    """
    asset_hints = request_json.get("assetHintsSummary") or request_json.get("assetHints") or ""
    if isinstance(request_json.get("input"), dict):
        asset_hints = request_json["input"].get("assetHintsSummary", asset_hints)
    linked_assets = []
    final_block = final_output.get("finalOutput") if isinstance(final_output, dict) else final_output
    if isinstance(final_block, dict):
        linked_assets = final_block.get("linkedAssets") or final_block.get("assets") or []
    return {"assetHintsSummary": asset_hints, "linkedAssets": linked_assets}


def _ensure_notes_file() -> List[str]:
    """
    Load the notes file (70_notes.md) as a mutable list of lines, creating it if
    missing. Storing lines keeps the Markdown structure easy to tweak.
    """
    if not NOTES_PATH.exists():
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTES_PATH.write_text("# Dev run annotations (70_notes.md)\n\n", encoding="utf-8")
    return NOTES_PATH.read_text(encoding="utf-8").splitlines()


def _append_note_for_request(request_id: str, note: str) -> Dict[str, str]:
    """
    Append a timestamped note under the requestId heading inside 70_notes.md.

    The file stays Markdown so humans can read/edit it, while we keep the write
    logic deterministic for the UI.
    """
    clean_note = " ".join(str(note).split())
    timestamp = datetime.utcnow().isoformat() + "Z"
    entry_line = f"- {timestamp}: {clean_note}"
    header_line = f"## {request_id}"

    lines = _ensure_notes_file()

    # Find or create the section for this requestId.
    if header_line not in lines:
        if lines and lines[-1].strip():
            lines.append("")  # keep a blank line before new section
        lines.extend([header_line, entry_line, ""])
    else:
        insert_at = lines.index(header_line) + 1
        # Skip past any existing bullet lines within this section.
        while insert_at < len(lines) and not lines[insert_at].startswith("## "):
            insert_at += 1
        lines.insert(insert_at, entry_line)

    NOTES_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {"timestamp": timestamp, "text": clean_note}


def _read_notes() -> Dict[str, List[Dict[str, str]]]:
    """
    Parse 70_notes.md into a dict keyed by requestId.
    """
    if not NOTES_PATH.exists():
        return {}
    notes: Dict[str, List[Dict[str, str]]] = {}
    current_request: Optional[str] = None
    for raw_line in NOTES_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current_request = line[3:].strip()
            continue
        if current_request and line.startswith("- "):
            payload = line[2:]
            if ": " in payload:
                ts, text = payload.split(": ", 1)
            else:
                ts, text = "", payload
            notes.setdefault(current_request, []).append({"timestamp": ts, "text": text})
    return notes


def _read_notes_for_request(request_id: str) -> List[Dict[str, str]]:
    """Return stored notes for a specific requestId."""
    all_notes = _read_notes()
    return all_notes.get(request_id, [])


def _enqueue_probe_plan(
    *,
    request_id: str,
    date_bucket: str,
    draft_text: str,
    missing_fields: List[str],
    follow_up_questions: List[str],
    tags: List[str],
) -> Dict[str, Any]:
    """
    Append a ProbePlan draft entry to a JSONL review queue for later triage.
    """
    PROBE_PLAN_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "requestId": request_id,
        "dateBucket": date_bucket,
        "draftText": draft_text,
        "missingFields": missing_fields,
        "followUpQuestions": follow_up_questions,
        "tags": tags,
        "createdAt": datetime.utcnow().isoformat() + "Z",
        "kind": "probePlanDraft",
    }
    with PROBE_PLAN_QUEUE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False))
        handle.write("\n")
    return entry


def _load_probe_plans_for_request(request_id: str) -> List[Dict[str, Any]]:
    """
    Read any ProbePlan draft entries for this requestId from the review queue.
    """
    if not PROBE_PLAN_QUEUE_PATH.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in PROBE_PLAN_QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("requestId") == request_id:
            entries.append(payload)
    return entries


def _load_run_detail(date_bucket: str, request_id: str) -> Dict[str, Any]:
    """
    Load all relevant files for a single bundle and derive UI-friendly slices.
    """
    run_path = LOG_ROOT / date_bucket / request_id
    index_json = _read_json_if_exists(run_path / "index.json")
    # Request payload may be named 00_request.json in some schemas; fall back to openai request.
    request_json = _read_json_if_exists(run_path / "00_request.json")
    if not request_json:
        request_json = _read_json_if_exists(run_path / "06_openai_request.json")
    response_meta = _read_json_if_exists(run_path / "21_openai_response_meta.json")
    trace_raw = _read_json_if_exists(run_path / "41_trace_raw.json")
    trace_sanitized = _read_json_if_exists(run_path / "42_trace_sanitized.json")
    trace_final = _read_json_if_exists(run_path / "43_trace_final.json")
    sanitize_reasons = _read_json_if_exists(run_path / "44_sanitize_reasons.json")
    parsed_output = _read_json_if_exists(run_path / "30_parsed_output.json")
    final_output = _read_json_if_exists(run_path / "50_final_output.json")
    system_prompt_text = _read_text_if_exists(run_path / "05_system_prompt.txt")

    transcript = _extract_transcript(request_json, trace_final)
    evidence_map = _extract_evidence_map(trace_final, sanitize_reasons)
    assets = _extract_assets(request_json, final_output or trace_final)

    missing_count = _compute_missing_count(trace_final, sanitize_reasons)

    return {
        "requestId": request_id,
        "dateBucket": date_bucket,
        "index": index_json,
        "request": request_json,
        "responseMeta": response_meta,
        "traceRaw": trace_raw,
        "traceSanitized": trace_sanitized,
        "traceFinal": trace_final,
        "sanitizeReasons": sanitize_reasons,
        "parsedOutput": parsed_output,
        "finalOutput": final_output,
        "systemPrompt": system_prompt_text,
        "transcript": transcript,
        "evidenceMap": evidence_map,
        "assets": assets,
        "missingCount": missing_count,
        "tokenCount": _token_count(response_meta),
        "tags": _extract_tags(request_json, index_json),
        "notes": _read_notes_for_request(request_id),
        "probePlanDrafts": _load_probe_plans_for_request(request_id),
    }


def _diff_extraction(parsed_output: Dict[str, Any], final_output: Dict[str, Any]) -> str:
    """
    Produce a small JSON diff string between parsed and final extractionTrace.
    """
    import difflib

    parsed_trace = json.dumps(parsed_output.get("extractionTrace", {}), ensure_ascii=False, indent=2, sort_keys=True)
    final_trace = json.dumps(final_output.get("extractionTrace", {}), ensure_ascii=False, indent=2, sort_keys=True)
    diff_lines = difflib.unified_diff(
        parsed_trace.splitlines(), final_trace.splitlines(), fromfile="30_parsed_output", tofile="50_final_output"
    )
    return "\n".join(diff_lines)


def _build_runs_inventory() -> List[Dict[str, Any]]:
    """
    Collect high-level metadata for the list view filters.
    """
    runs: List[Dict[str, Any]] = []
    for date_bucket, run_path in _gather_bundle_paths():
        request_id = run_path.name
        index_json = _read_json_if_exists(run_path / "index.json")
        final_trace = _read_json_if_exists(run_path / "43_trace_final.json")
        sanitize_reasons = _read_json_if_exists(run_path / "44_sanitize_reasons.json")
        response_meta = _read_json_if_exists(run_path / "21_openai_response_meta.json")

        missing = _compute_missing_count(final_trace, sanitize_reasons)
        tokens = _token_count(response_meta)
        tags = _extract_tags(_read_json_if_exists(run_path / "00_request.json"), index_json)
        created_at = index_json.get("createdAt") if isinstance(index_json, dict) else None

        runs.append(
            {
                "requestId": request_id,
                "dateBucket": date_bucket,
                "createdAt": created_at,
                "promptVersion": index_json.get("promptVersion") if isinstance(index_json, dict) else None,
                "schemaVersion": index_json.get("schemaVersion") if isinstance(index_json, dict) else None,
                "missingCount": missing,
                "tokenCount": tokens,
                "tags": tags,
                "path": f"/dev/logs/{date_bucket}/{request_id}",
                # durationMs is not present in the sample schema; keep slot for future data.
                "durationMs": index_json.get("durationMs") if isinstance(index_json, dict) else None,
            }
        )
    return runs


def _html_shell() -> str:
    """
    Return the single-page HTML (vanilla JS) used for both list and detail views.
    The JS detects whether the path looks like /dev/logs/<date>/<id> and fetches
    the appropriate API endpoint.
    """
    return r"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Developer Log Viewer</title>
    <style>
      :root { color-scheme: light; --bg:#f5f7fb; --card:#fff; --text:#1c1f2b; --muted:#6c7480; --accent:#0f6ad8; }
      body { margin:0; font-family: system-ui, -apple-system, sans-serif; background:var(--bg); color:var(--text); }
      header { padding:16px 24px; background:#0b2340; color:#e8f0ff; }
      h1 { margin:0; font-size:20px; }
      main { padding:20px; max-width:1200px; margin:0 auto; }
      .card { background:var(--card); border-radius:12px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
      table { width:100%; border-collapse:collapse; }
      th, td { text-align:left; padding:8px; border-bottom:1px solid #e3e7ef; }
      th { color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:0.02em; }
      .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; background:#e8f0ff; color:#0b2340; border-radius:999px; font-size:12px; }
      .pill-button { display:inline-flex; align-items:center; gap:6px; padding:8px 12px; background:#0f6ad8; color:#fff; border:none; border-radius:10px; font-weight:600; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
      .pill-button:disabled { background:#9bb8e8; cursor:not-allowed; }
      .tabs { display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }
      .tab { padding:10px 14px; border:1px solid #d0d7e5; border-radius:10px; background:#fff; cursor:pointer; }
      .tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
      .tab-content { display:none; }
      .tab-content.active { display:block; }
      pre { background:#0b2340; color:#e8f0ff; padding:12px; border-radius:10px; overflow:auto; }
      .filters { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
      .filters label { font-size:13px; color:var(--muted); display:flex; flex-direction:column; gap:4px; }
      input[type=\"number\"], input[type=\"text\"] { padding:8px 10px; border-radius:8px; border:1px solid #cbd4e3; min-width:120px; }
      .transcript-row.highlight { background:#fff6d6; }
      .grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
      .callout { padding:12px; background:#f0f4ff; border:1px solid #d3dffb; border-radius:10px; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
      .small { color:var(--muted); font-size:12px; }
      textarea { width:100%; min-height:90px; padding:10px; border-radius:10px; border:1px solid #cbd4e3; resize:vertical; font-family:inherit; }
      .stack { display:flex; flex-direction:column; gap:8px; }
      .meta-actions { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
      .pill-button.secondary { background:#16345c; }
      .muted { color:var(--muted); }
    </style>
  </head>
  <body>
    <header>
      <h1>Developer Log Viewer</h1>
      <div class="small">Investigate each request: input → prompt → model output → UI patch.</div>
    </header>
    <main id="app"></main>

    <template id="list-view">
      <section class="card">
        <div class="filters">
          <label>missingCount &gt; <input id="filter-missing" type="number" value="0" min="0" /></label>
          <label>durationMs &gt; <input id="filter-duration" type="number" value="0" min="0" /></label>
          <label>tokens &gt; <input id="filter-tokens" type="number" value="0" min="0" /></label>
          <label>tag contains <input id="filter-tag" type="text" placeholder="safety, rx, ..." /></label>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Request</th>
                <th>Created</th>
                <th>Prompt</th>
                <th>Missing</th>
                <th>Tokens</th>
                <th>Tags</th>
              </tr>
            </thead>
            <tbody id="runs-body"></tbody>
          </table>
        </div>
      </section>
    </template>

    <template id="detail-view">
      <section class="card" id="meta"></section>
      <div class="tabs" id="tabs"></div>
      <section class="card" id="tab-panels"></section>
    </template>

    <script>
      const app = document.getElementById('app');

      const pathParts = window.location.pathname.split('/').filter(Boolean);
      const isDetail = pathParts.length >= 4 && pathParts[0] === 'dev' && pathParts[1] === 'logs';

      function renderList(runs) {
        const view = document.getElementById('list-view').content.cloneNode(true);
        const body = view.querySelector('#runs-body');
        const filters = {
          missing: view.querySelector('#filter-missing'),
          duration: view.querySelector('#filter-duration'),
          tokens: view.querySelector('#filter-tokens'),
          tag: view.querySelector('#filter-tag'),
        };

        function applyFilters() {
          const missingMin = Number(filters.missing.value || 0);
          const durationMin = Number(filters.duration.value || 0);
          const tokensMin = Number(filters.tokens.value || 0);
          const tagText = filters.tag.value.toLowerCase();
          body.innerHTML = '';
          runs
            .filter((run) => run.missingCount > missingMin)
            .filter((run) => (run.durationMs || 0) > durationMin)
            .filter((run) => (run.tokenCount || 0) > tokensMin)
            .filter((run) => {
              if (!tagText) return true;
              return (run.tags || []).some((t) => String(t).toLowerCase().includes(tagText));
            })
            .forEach((run) => {
              const tr = document.createElement('tr');
              const tags = (run.tags || []).map((t) => `<span class=\"pill\">${t}</span>`).join(' ');
              tr.innerHTML = `
                <td><a href=\"${run.path}\"><strong>${run.requestId}</strong></a><div class=\"small mono\">${run.path}</div></td>
                <td>${run.createdAt || ''}</td>
                <td>${run.promptVersion || ''}</td>
                <td>${run.missingCount}</td>
                <td>${run.tokenCount ?? ''}</td>
                <td>${tags}</td>
              `;
              body.appendChild(tr);
            });
        }

        Object.values(filters).forEach((input) => input.addEventListener('input', applyFilters));
        applyFilters();
        app.innerHTML = '';
        app.appendChild(view);
      }

      function renderTranscript(transcript) {
        if (!transcript?.length) return '<div class=\"small\">No transcript found.</div>';
        const rows = transcript
          .map(
            (turn) => `
              <tr id=\"turn-${turn.idx}\" class=\"transcript-row\">
                <td class=\"mono\">#${turn.idx}</td>
                <td>${turn.speaker}</td>
                <td>${turn.text}</td>
              </tr>
            `
          )
          .join('');
        return `<table><thead><tr><th>#</th><th>Speaker</th><th>Text</th></tr></thead><tbody>${rows}</tbody></table>`;
      }

      function renderEvidence(evidence, transcript) {
        if (!evidence?.length) return '<div class=\"small\">No evidence map available.</div>';
        const rows = evidence
          .map((row) => {
            const targetId = row.turnIndex != null ? `turn-${row.turnIndex}` : null;
            return `
              <tr data-target=\"${targetId || ''}\" class=\"evidence-row\">
                <td class=\"mono\">${row.field}</td>
                <td>${row.status || ''}</td>
                <td>${row.turnIndex ?? ''}</td>
                <td>${row.snippet || row.value || ''}</td>
                <td>${row.reason || ''}</td>
              </tr>
            `;
          })
          .join('');
        const table = document.createElement('div');
        table.innerHTML = `<table><thead><tr><th>Field</th><th>Status</th><th>Turn</th><th>Snippet / Value</th><th>Reason</th></tr></thead><tbody>${rows}</tbody></table>`;
        table.querySelectorAll('.evidence-row').forEach((row) => {
          const target = row.dataset.target;
          if (!target) return;
          row.style.cursor = 'pointer';
          row.addEventListener('click', () => {
            document.querySelectorAll('.transcript-row').forEach((tr) => tr.classList.remove('highlight'));
            const el = document.getElementById(target);
            if (el) {
              el.classList.add('highlight');
              el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          });
        });
        return table.innerHTML;
      }

      function renderAssets(assets) {
        return `
          <div class=\"grid\">
            <div class=\"callout\"><strong>Asset Hints (input)</strong><div>${assets.assetHintsSummary || '—'}</div></div>
            <div class=\"callout\"><strong>Linked Assets (output)</strong><pre>${JSON.stringify(assets.linkedAssets || [], null, 2)}</pre></div>
          </div>
        `;
      }

      function renderDiff(diffText) {
        if (!diffText) return '<div class=\"small\">No diff available.</div>';
        return `<pre class=\"mono\">${diffText.replace(/</g, '&lt;')}</pre>`;
      }

      function renderRequestTab(data) {
        const payload = data.request && Object.keys(data.request).length ? data.request : data.traceRaw;
        return `
          <div class=\"grid\">
            <div>
              <div class=\"small\">System Prompt</div>
              <pre>${(data.systemPrompt || '').replace(/</g, '&lt;')}</pre>
            </div>
            <div>
              <div class=\"small\">Request Payload</div>
              <pre>${JSON.stringify(payload || {}, null, 2).replace(/</g, '&lt;')}</pre>
            </div>
          </div>
        `;
      }

      function renderNoteHistory(notes) {
        if (!notes?.length) return '<div class=\"small muted\">No saved notes for this request yet.</div>';
        return notes
          .map(
            (note) => `
              <div class=\"callout\" style=\"margin-bottom:6px;\">
                <div class=\"small mono\">${note.timestamp || ''}</div>
                <div>${note.text || ''}</div>
              </div>
            `
          )
          .join('');
      }

      function collectArray(payload, key) {
        const values = [];
        if (!payload || typeof payload !== 'object') return values;
        if (Array.isArray(payload[key])) values.push(...payload[key]);
        if (payload.finalOutput && Array.isArray(payload.finalOutput[key])) values.push(...payload.finalOutput[key]);
        return values;
      }

      function extractProbePlanInputs(data) {
        const sources = [data.finalOutput, data.traceFinal, data.parsedOutput];
        const missingFields = [];
        const followUpQuestions = [];
        sources.forEach((payload) => {
          missingFields.push(...collectArray(payload, 'missingFields'));
          followUpQuestions.push(...collectArray(payload, 'followUpQuestions'));
        });
        const dedupe = (arr) => Array.from(new Set(arr.filter(Boolean).map((item) => String(item))));
        return { missingFields: dedupe(missingFields), followUpQuestions: dedupe(followUpQuestions) };
      }

      function buildProbePlanDraft(data) {
        const { missingFields, followUpQuestions } = extractProbePlanInputs(data);
        const generatedAt = new Date().toISOString();
        const lines = [
          `# ProbePlan draft for ${data.requestId}`,
          `- generatedAt: ${generatedAt}`,
          `- missingCount: ${data.missingCount ?? 0}`,
          '',
          '## missingFields',
          ...(missingFields.length ? missingFields.map((field) => `- ${field}`) : ['- (none detected)']),
          '',
          '## followUpQuestions',
          ...(followUpQuestions.length ? followUpQuestions.map((q) => `- ${q}`) : ['- (none detected)']),
          '',
          '## probe actions',
          '- Draft probes to close the gaps above (turn snippets into concrete asks).',
          '- Link findings back to this requestId when promoting to assets.',
        ];
        return { draftText: lines.join('\\n'), missingFields, followUpQuestions, generatedAt };
      }

      function loadLocalProbePlan(requestId) {
        const raw = localStorage.getItem(`probePlanDraft:${requestId}`);
        if (!raw) return null;
        try {
          return JSON.parse(raw);
        } catch {
          return null;
        }
      }

      function persistLocalProbePlan(requestId, draft) {
        localStorage.setItem(`probePlanDraft:${requestId}`, JSON.stringify(draft));
      }

      function renderProbePreview(draft) {
        if (!draft) return '<div class=\"small muted\">No ProbePlan draft yet. Click “ProbePlan下書きを作る”.</div>';
        const summary = draft.draftText || '';
        return `<pre class=\"mono\" style=\"max-height:220px; overflow:auto; white-space:pre-wrap;\">${summary.replace(/</g, '&lt;')}</pre>`;
      }

      function bindNotes(root, data, requestId) {
        const noteInput = root.querySelector('#note-input');
        const noteStatus = root.querySelector('#note-status');
        const noteHistory = root.querySelector('#note-history');
        const noteButton = root.querySelector('#save-note');
        if (!noteInput || !noteStatus || !noteHistory || !noteButton) return;

        const draftKey = `noteDraft:${requestId}`;
        const cachedDraft = localStorage.getItem(draftKey);
        if (cachedDraft) noteInput.value = cachedDraft;

        noteHistory.innerHTML = renderNoteHistory(data.notes);

        noteInput.addEventListener('input', () => {
          localStorage.setItem(draftKey, noteInput.value);
        });

        noteButton.addEventListener('click', async () => {
          const note = noteInput.value.trim();
          if (!note) {
            noteStatus.textContent = 'Enter a note before saving.';
            return;
          }
          noteButton.disabled = true;
          noteStatus.textContent = 'Saving to 70_notes.md...';
          try {
            const res = await fetch('/dev/logs/api/note', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ requestId, note }),
            });
            if (!res.ok) throw new Error('save failed');
            const payload = await res.json();
            noteHistory.innerHTML = renderNoteHistory(payload.notes);
            noteStatus.textContent = 'Saved to 70_notes.md (dev-only).';
            localStorage.removeItem(draftKey);
            noteInput.value = '';
          } catch (error) {
            noteStatus.textContent = 'Save failed. Ensure NODE_ENV=development and DAL_LOG_UI=true.';
          } finally {
            noteButton.disabled = false;
          }
        });
      }

      function bindProbePlan(root, data, requestId, dateBucket) {
        const preview = root.querySelector('#probe-plan-preview');
        const status = root.querySelector('#probe-plan-status');
        const buildBtn = root.querySelector('#probe-plan-btn');
        const saveBtn = root.querySelector('#save-probe-plan');

        const serverDrafts = Array.isArray(data.probePlanDrafts) ? data.probePlanDrafts : [];
        const latestServer = serverDrafts[serverDrafts.length - 1] || null;
        const initialDraft = loadLocalProbePlan(requestId) || latestServer;

        if (preview) {
          preview.innerHTML = renderProbePreview(initialDraft);
        }

        const buildDraft = () => {
          const draft = buildProbePlanDraft(data);
          persistLocalProbePlan(requestId, draft);
          if (preview) preview.innerHTML = renderProbePreview(draft);
          if (status) status.textContent = 'Draft updated locally from missingFields + followUpQuestions.';
          return draft;
        };

        if (buildBtn) {
          buildBtn.addEventListener('click', () => {
            buildDraft();
          });
        }

        if (saveBtn) {
          saveBtn.addEventListener('click', async () => {
            const draft = loadLocalProbePlan(requestId) || buildDraft();
            saveBtn.disabled = true;
            if (status) status.textContent = 'Queueing draft to review...';
            try {
              const res = await fetch('/dev/logs/api/probe-plan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  requestId,
                  dateBucket,
                  draftText: draft.draftText,
                  missingFields: draft.missingFields,
                  followUpQuestions: draft.followUpQuestions,
                  tags: data.tags || [],
                }),
              });
              if (!res.ok) throw new Error('queue failed');
              const payload = await res.json();
              if (payload.queued) {
                persistLocalProbePlan(requestId, payload.queued);
                if (preview) preview.innerHTML = renderProbePreview(payload.queued);
              }
              if (status) status.textContent = 'Queued for review and saved locally.';
            } catch (error) {
              if (status) status.textContent = 'Queue failed. Dev-only endpoint requires NODE_ENV=development + DAL_LOG_UI=true.';
            } finally {
              saveBtn.disabled = false;
            }
          });
        }
      }

      function renderMeta(data) {
        const probeDrafts = Array.isArray(data.probePlanDrafts) ? data.probePlanDrafts : [];
        const latestServerDraft = probeDrafts[probeDrafts.length - 1] || null;
        return `
          <div class=\"stack\">
            <div style=\"display:flex; align-items:center; gap:12px; flex-wrap:wrap;\">
              <div>
                <div><strong>${data.requestId}</strong> • ${data.index.schemaVersion || ''}</div>
                <div class=\"small mono\">${data.dateBucket} / missingCount=${data.missingCount} / tokens=${data.tokenCount ?? 'n/a'}</div>
              </div>
              <div class=\"meta-actions\">
                <button id=\"download-bundle\" class=\"pill-button\">Download bundle</button>
                <button id=\"probe-plan-btn\" class=\"pill-button secondary\">ProbePlan下書きを作る</button>
              </div>
            </div>
            <div class=\"grid\">
              <div class=\"callout stack\">
                <div><strong>Annotations</strong> (70_notes.md / dev-only)</div>
                <textarea id=\"note-input\" placeholder=\"Write a quick insight, asset idea, or follow-up for this run.\"></textarea>
                <div class=\"meta-actions\">
                  <button id=\"save-note\" class=\"pill-button\">Save note</button>
                  <span id=\"note-status\" class=\"small muted\"></span>
                </div>
                <div id=\"note-history\">${renderNoteHistory(data.notes)}</div>
              </div>
              <div class=\"callout stack\">
                <div><strong>ProbePlan draft</strong> (missingFields + followUpQuestions)</div>
                <div id=\"probe-plan-preview\">${renderProbePreview(latestServerDraft || loadLocalProbePlan(data.requestId))}</div>
                <div class=\"meta-actions\">
                  <button id=\"save-probe-plan\" class=\"pill-button\">Queue to review</button>
                  <span id=\"probe-plan-status\" class=\"small muted\"></span>
                </div>
                <div class=\"small muted\">Drafts are cached locally and appended to the review queue when saved.</div>
              </div>
            </div>
          </div>
        `;
      }

      async function renderDetail() {
        const [_, __, ___, dateBucket, requestId] = pathParts;
        const res = await fetch(`/dev/logs/api/run/${dateBucket}/${requestId}`);
        const data = await res.json();

        const detail = document.getElementById('detail-view').content.cloneNode(true);
        detail.querySelector('#meta').innerHTML = renderMeta(data);
        const tabs = [
          { id: 'request', label: 'Request', content: renderRequestTab(data) },
          { id: 'transcript', label: 'Transcript', content: renderTranscript(data.transcript) },
          { id: 'evidence', label: 'Evidence Map', content: renderEvidence(data.evidenceMap, data.transcript) },
          { id: 'assets', label: 'Assets', content: renderAssets(data.assets) },
          { id: 'diff', label: 'Diff', content: renderDiff(data.diffExtraction || '') },
        ];

        const tabsEl = detail.querySelector('#tabs');
        const panelsEl = detail.querySelector('#tab-panels');

        tabs.forEach((tab, idx) => {
          const btn = document.createElement('button');
          btn.className = 'tab' + (idx === 0 ? ' active' : '');
          btn.textContent = tab.label;
          btn.dataset.target = tab.id;
          tabsEl.appendChild(btn);

          const panel = document.createElement('div');
          panel.id = tab.id;
          panel.className = 'tab-content' + (idx === 0 ? ' active' : '');
          panel.innerHTML = tab.content;
          panelsEl.appendChild(panel);
        });

        tabsEl.addEventListener('click', (event) => {
          if (!(event.target instanceof HTMLElement)) return;
          if (!event.target.dataset.target) return;
          const target = event.target.dataset.target;
          document.querySelectorAll('.tab').forEach((tab) => tab.classList.remove('active'));
          event.target.classList.add('active');
          panelsEl.querySelectorAll('.tab-content').forEach((panel) => {
            panel.classList.toggle('active', panel.id === target);
          });
        });

        app.innerHTML = '';
        app.appendChild(detail);

        const downloadButton = detail.querySelector('#download-bundle');
        if (downloadButton) {
          downloadButton.addEventListener('click', async () => {
            downloadButton.disabled = true;
            try {
              const response = await fetch(`/api/dev/trace-bundle/download?requestId=${encodeURIComponent(requestId)}`);
              if (!response.ok) {
                throw new Error('Download failed');
              }
              const blob = await response.blob();
              const url = URL.createObjectURL(blob);
              const link = document.createElement('a');
              link.href = url;
              link.download = `${requestId}.zip`;
              document.body.appendChild(link);
              link.click();
              link.remove();
              URL.revokeObjectURL(url);
            } catch (error) {
              alert('Download unavailable (dev-only or bundle missing).');
            } finally {
              downloadButton.disabled = false;
            }
          });
        }

        bindNotes(detail, data, requestId);
        bindProbePlan(detail, data, requestId, dateBucket);
      }

      async function boot() {
        if (isDetail) {
          await renderDetail();
        } else {
          const res = await fetch('/dev/logs/api/runs');
          const payload = await res.json();
          renderList(payload.runs || []);
        }
      }
      boot();
    </script>
  </body>
</html>
"""


class DevLogViewerHandler(SimpleHTTPRequestHandler):
    """
    Custom request handler that serves both static HTML and JSON APIs for the
    developer log viewer. Everything runs from a single file to keep the setup
    simple for new contributors.
    """

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Write a JSON response with the provided status code."""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        """Serve the SPA shell HTML."""
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/dev/trace-bundle/download"):
            if not _dev_bundle_download_enabled():
                # Return a soft 404 to avoid exposing the endpoint outside dev.
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return

            params = urllib.parse.parse_qs(parsed.query)
            request_id_values = params.get("requestId")
            if not request_id_values or not request_id_values[0]:
                self._send_json({"error": "requestId is required"}, HTTPStatus.BAD_REQUEST)
                return

            request_id = request_id_values[0]
            bundle = _find_bundle_by_request_id(request_id)
            if not bundle:
                self._send_json({"error": "bundle not found"}, HTTPStatus.NOT_FOUND)
                return

            _, bundle_path = bundle
            payload = _zip_bundle(bundle_path)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{request_id}.zip"')
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path.startswith("/dev/logs/api/runs"):
            runs = _build_runs_inventory()
            self._send_json({"runs": runs})
            return

        if path.startswith("/dev/logs/api/run/"):
            segments = [seg for seg in path.split("/") if seg]
            # Expect ["dev", "logs", "api", "run", "<date>", "<requestId>"]
            if len(segments) < 6:
                self._send_json({"error": "expected /dev/logs/api/run/<date>/<requestId>"}, HTTPStatus.BAD_REQUEST)
                return
            date_bucket, request_id = segments[4], segments[5]
            detail = _load_run_detail(date_bucket, request_id)
            parsed_output = detail.get("parsedOutput") or {}
            final_output = detail.get("finalOutput") or {}
            if parsed_output or final_output:
                detail["diffExtraction"] = _diff_extraction(parsed_output, final_output)
            else:
                detail["diffExtraction"] = ""
            self._send_json(detail)
            return

        if path.startswith("/dev/logs"):
            # Serve the SPA shell for both list and detail routes.
            self._send_html(_html_shell())
            return

        # Fallback to default behavior (e.g., serve files if needed).
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/dev/logs/api/note":
            if not _dev_mutation_enabled():
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return

            content_length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON body"}, HTTPStatus.BAD_REQUEST)
                return

            request_id = payload.get("requestId")
            note_text = payload.get("note")
            if not request_id or not isinstance(request_id, str):
                self._send_json({"error": "requestId is required"}, HTTPStatus.BAD_REQUEST)
                return
            if not note_text or not isinstance(note_text, str):
                self._send_json({"error": "note is required"}, HTTPStatus.BAD_REQUEST)
                return

            note_entry = _append_note_for_request(request_id, note_text)
            notes = _read_notes_for_request(request_id)
            self._send_json({"saved": note_entry, "notes": notes})
            return

        if path == "/dev/logs/api/probe-plan":
            if not _dev_mutation_enabled():
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return

            content_length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON body"}, HTTPStatus.BAD_REQUEST)
                return

            request_id = payload.get("requestId")
            date_bucket = payload.get("dateBucket") or _today_str()
            draft_text = payload.get("draftText") or ""
            missing_fields = payload.get("missingFields") or []
            follow_up_questions = payload.get("followUpQuestions") or []
            tags = payload.get("tags") or []

            if not request_id or not isinstance(request_id, str):
                self._send_json({"error": "requestId is required"}, HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(draft_text, str) or not draft_text.strip():
                self._send_json({"error": "draftText is required"}, HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(missing_fields, list) or not isinstance(follow_up_questions, list):
                self._send_json({"error": "missingFields and followUpQuestions must be lists"}, HTTPStatus.BAD_REQUEST)
                return
            if not isinstance(tags, list):
                self._send_json({"error": "tags must be a list"}, HTTPStatus.BAD_REQUEST)
                return

            entry = _enqueue_probe_plan(
                request_id=request_id,
                date_bucket=str(date_bucket),
                draft_text=draft_text,
                missing_fields=[str(item) for item in missing_fields],
                follow_up_questions=[str(item) for item in follow_up_questions],
                tags=[str(item) for item in tags],
            )
            self._send_json({"queued": entry, "existing": _load_probe_plans_for_request(request_id)})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")


def main() -> None:
    """CLI entrypoint for launching the viewer server."""
    parser = argparse.ArgumentParser(description="Developer log viewer server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), DevLogViewerHandler)
    print(f"Dev Log Viewer running at http://localhost:{args.port}/dev/logs")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down viewer...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
