"""Pipecat tool definitions for the local voice pipeline.

The big one is `work` — it writes a task file to the Sutando workspace
(tasks/) and polls results/ for the response. That's the same bridge bodhi
uses, so every Sutando capability (email, calendar, calls, web search,
recording, …) is reachable from voice via this single function. The core
(Claude Code session running watch-tasks-stream.sh) does the actual work.

Native Python tools handle a handful of low-latency operations that don't
need a round-trip to core: time, notes, recent_context, summon (Zoom).

Why no Node sidecar: `work` fans out to core, which already imports
src/inline-tools.ts via the existing voice-agent.ts → bodhi → Gemini path.
We don't have to re-implement those 30 tools in Python; we just need to
queue them as tasks for core to handle, which is exactly what bodhi does.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, LLMService


# --- workspace resolution (matches Sutando convention from CLAUDE.md) ---


def _repo_root() -> Path:
    """The Sutando repo root: skills/local-voice-pipeline/scripts/ → up 3."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _resolve_tasks_dir() -> Path:
    """Resolve where to write task files.

    Must match the watcher (src/watch-tasks-stream.sh) — if pipeline.py
    writes to a different dir than the watcher polls, the result watcher
    and core never see the task. Resolution order:

    1. ``$SUTANDO_TASKS_DIR`` (explicit override — same name the watcher
       would accept via positional arg)
    2. ``$SUTANDO_WORKSPACE/tasks`` if set
    3. Repo's ``tasks/`` if it exists AND has tasks already (mid-migration
       machines still write here; matches the user's current watcher
       invocation `bash src/watch-tasks-stream.sh <repo>/tasks`)
    4. Canonical default ``~/.sutando/workspace/tasks``
    """
    env = os.environ.get("SUTANDO_TASKS_DIR")
    if env:
        return Path(env).expanduser()
    ws_env = os.environ.get("SUTANDO_WORKSPACE")
    if ws_env:
        return Path(ws_env).expanduser() / "tasks"
    repo_tasks = _repo_root() / "tasks"
    if repo_tasks.exists() and any(repo_tasks.glob("*.txt")):
        return repo_tasks
    return Path.home() / ".sutando" / "workspace" / "tasks"


def _resolve_results_dir() -> Path:
    """Same logic as tasks/, applied to results/."""
    env = os.environ.get("SUTANDO_RESULTS_DIR")
    if env:
        return Path(env).expanduser()
    return _resolve_tasks_dir().parent / "results"


TASKS_DIR = _resolve_tasks_dir()
RESULTS_DIR = _resolve_results_dir()
WORKSPACE = TASKS_DIR.parent
NOTES_DIR = WORKSPACE / "notes"
STATE_DIR = WORKSPACE / "state"
VOICE_SESSION_CONTEXT = STATE_DIR / "voice-session-context.json"


# ===========================================================================
# work — fan-out into core
# ===========================================================================


WORK_SCHEMA = FunctionSchema(
    name="work",
    description=(
        "Do the work. Call this for anything beyond simple greetings — "
        "questions, actions, research, writing, translation, file changes, "
        "system queries, explanations, analysis. This is how Sutando thinks "
        "and acts. Results are spoken back when ready. While the task is "
        "running, tell the user you're working on it; do NOT claim it's "
        "done."
    ),
    properties={
        "task": {
            "type": "string",
            "description": "Full natural-language description of the task.",
        },
        "timeout_minutes": {
            "type": "number",
            "description": (
                "How long to wait before timing out, in minutes. Default 10. "
                "Pass 0 for no timeout. Capped at 360."
            ),
        },
    },
    required=["task"],
)


async def work_handler(params: FunctionCallParams) -> None:
    """Write a task file and poll for the result.

    Mirrors the contract in src/task-bridge.ts:workTool — same file
    layout, same priority, same source/channel_id, so the result watcher
    and dashboard treat pipecat-voice tasks identically to bodhi-voice
    tasks.
    """
    args = params.arguments or {}
    task_text = str(args.get("task", "")).strip()
    if not task_text:
        await params.result_callback({"status": "error", "error": "empty task"})
        return

    timeout_min = args.get("timeout_minutes")
    if not isinstance(timeout_min, (int, float)) or timeout_min < 0:
        timeout_min = 10
    elif timeout_min == 0:
        timeout_min = 360  # cap; "no timeout" still gets a hard ceiling
    timeout_min = min(float(timeout_min), 360.0)
    timeout_secs = timeout_min * 60

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    task_id = f"task-{int(time.time() * 1000)}"
    owner_id = os.environ.get("SUTANDO_DM_OWNER_ID", "voice-local")
    ts_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    content = (
        f"id: {task_id}\n"
        f"timestamp: {ts_iso}\n"
        f"task: {task_text}\n"
        f"source: voice\n"
        f"channel_id: local-voice\n"
        f"user_id: {owner_id}\n"
        f"access_tier: owner\n"
        f"priority: urgent\n"
    )
    task_path = TASKS_DIR / f"{task_id}.txt"
    task_path.write_text(content)
    logger.info(f"work: queued {task_id} — {task_text[:80]}")

    # Poll for result file. Start with short interval (responsive for fast
    # tasks), back off after ~30s (long-running tasks don't need 5Hz polling).
    result_path = RESULTS_DIR / f"{task_id}.txt"
    deadline = time.time() + timeout_secs
    interval = 0.2
    while time.time() < deadline:
        if result_path.exists():
            try:
                body = result_path.read_text().strip()
            except Exception as e:
                body = f"(failed to read result: {e})"
            # Strip the bridge protocol markers — those are for the
            # Discord/Telegram bridges, not for the LLM to know about.
            for marker in ("[deduped:", "[no-send]", "[REPLIED]"):
                if body.startswith(marker):
                    body = body.split("\n", 1)[1] if "\n" in body else ""
                    break
            await params.result_callback({"status": "done", "result": body})
            logger.info(f"work: {task_id} done ({len(body)} chars)")
            return
        await asyncio.sleep(interval)
        if time.time() - (deadline - timeout_secs) > 30 and interval < 1.0:
            interval = 1.0

    logger.warning(f"work: {task_id} timed out after {timeout_min}min")
    await params.result_callback(
        {
            "status": "timeout",
            "result": (
                "The task is still running but I haven't heard back yet. "
                "Tell the user it's taking longer than expected and they "
                "can ask for an update later."
            ),
        }
    )


# ===========================================================================
# get_current_time — native, instant
# ===========================================================================


CURRENT_TIME_SCHEMA = FunctionSchema(
    name="get_current_time",
    description=(
        "Get the current local date and time. Call this when the user asks "
        "what time it is or what today's date is."
    ),
    properties={},
    required=[],
)


async def current_time_handler(params: FunctionCallParams) -> None:
    now = datetime.now()
    await params.result_callback(
        {
            "status": "done",
            "result": now.strftime("It's %A, %B %d, %Y, %-I:%M %p."),
        }
    )


# ===========================================================================
# recent_context — read state/voice-session-context.json
# ===========================================================================


RECENT_CONTEXT_SCHEMA = FunctionSchema(
    name="recent_context",
    description=(
        "Recall recent durable decisions from the current Sutando session: "
        "active drafts, pending paste/clipboard actions, last task results. "
        "Call this when the user references something earlier that you "
        "might have forgotten (\"what was the post?\", \"what's pending?\")."
    ),
    properties={},
    required=[],
)


async def recent_context_handler(params: FunctionCallParams) -> None:
    if not VOICE_SESSION_CONTEXT.exists():
        await params.result_callback({"status": "done", "result": "No recent context."})
        return
    try:
        data = json.loads(VOICE_SESSION_CONTEXT.read_text())
    except Exception as e:
        await params.result_callback({"status": "error", "error": f"could not read context: {e}"})
        return
    parts: list[str] = []
    drafts = data.get("active_drafts") or []
    if drafts:
        parts.append("Active drafts: " + "; ".join(f"{d.get('name')} — {d.get('summary')}" for d in drafts[-3:]))
    pa = data.get("pending_action")
    if pa:
        parts.append(f"Pending action: {pa.get('kind')} — {pa.get('what')} at {pa.get('where')}")
    last = data.get("last_results") or []
    if last:
        parts.append("Recent results: " + "; ".join(r.get("subject", "") for r in last[-3:]))
    result = " | ".join(parts) if parts else "No recent context."
    await params.result_callback({"status": "done", "result": result})


# ===========================================================================
# save_note / read_note — notes/ directory
# ===========================================================================


SAVE_NOTE_SCHEMA = FunctionSchema(
    name="save_note",
    description=(
        "Save a note to the user's second brain. Use when they say "
        '"remember this", "take a note", "save this for later", or '
        "want to capture an idea, research finding, or bookmark. "
        "The title becomes the filename."
    ),
    properties={
        "title": {"type": "string", "description": "Short descriptive title for the note."},
        "content": {"type": "string", "description": "Body of the note."},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for the note (e.g. ['ideas', 'projects']).",
        },
    },
    required=["title", "content"],
)


def _slugify(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    out = []
    for ch in nfkd.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:60] or "untitled"


async def save_note_handler(params: FunctionCallParams) -> None:
    args = params.arguments or {}
    title = str(args.get("title", "")).strip()
    content = str(args.get("content", "")).strip()
    tags = args.get("tags") or []
    if not title or not content:
        await params.result_callback({"status": "error", "error": "title and content required"})
        return
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(title)
    path = NOTES_DIR / f"{slug}.md"
    # Don't clobber — if a note with that slug exists, append a suffix
    n = 2
    while path.exists():
        path = NOTES_DIR / f"{slug}-{n}.md"
        n += 1
    today = datetime.now().strftime("%Y-%m-%d")
    tags_yaml = "[" + ", ".join(tags) + "]" if tags else "[]"
    body = (
        f"---\n"
        f"title: {title}\n"
        f"date: {today}\n"
        f"tags: {tags_yaml}\n"
        f"---\n\n"
        f"{content}\n"
    )
    path.write_text(body)
    logger.info(f"save_note: {path}")
    await params.result_callback({"status": "done", "result": f"Saved as {path.name}."})


READ_NOTE_SCHEMA = FunctionSchema(
    name="read_note",
    description=(
        "Look up an existing note from the user's second brain. "
        "Pass a query (matches against title or body)."
    ),
    properties={"query": {"type": "string", "description": "Search term or note title."}},
    required=["query"],
)


async def read_note_handler(params: FunctionCallParams) -> None:
    args = params.arguments or {}
    query = str(args.get("query", "")).strip().lower()
    if not query:
        await params.result_callback({"status": "error", "error": "query required"})
        return
    if not NOTES_DIR.exists():
        await params.result_callback({"status": "done", "result": "No notes yet."})
        return
    # Score files: title-match wins, body-match is fallback
    candidates: list[tuple[int, Path]] = []
    for p in NOTES_DIR.glob("*.md"):
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        score = 0
        if query in p.stem.lower():
            score += 10
        if query in txt.lower():
            score += 1
        if score:
            candidates.append((score, p))
    if not candidates:
        await params.result_callback({"status": "done", "result": "No matching note."})
        return
    candidates.sort(key=lambda c: -c[0])
    best = candidates[0][1]
    body = best.read_text()
    # Strip YAML frontmatter for spoken output
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].strip()
    await params.result_callback({"status": "done", "result": body[:1500]})


# ===========================================================================
# summon — open Zoom + screen share + computer audio
# ===========================================================================


SUMMON_SCHEMA = FunctionSchema(
    name="summon",
    description=(
        "Summon Zoom with computer audio + screen share. Use when the user "
        "wants to start a Zoom call hands-free. Returns once the call is up."
    ),
    properties={},
    required=[],
)


async def summon_handler(params: FunctionCallParams) -> None:
    # Delegate to the inline-tool via the TS bootstrap that CLAUDE.md
    # documents. Subprocess-out so we get whatever logic the TS-side has
    # for muting, audio routing, etc. Run in a thread so we don't block
    # the pipeline event loop.
    cmd = [
        "npx",
        "tsx",
        "-e",
        (
            "import 'dotenv/config';"
            "import {summonTool} from './src/inline-tools.ts';"
            "summonTool.execute({}, null).then(r => console.log(JSON.stringify(r)));"
        ),
    ]
    cwd = WORKSPACE if (WORKSPACE / "src" / "inline-tools.ts").exists() else None
    if cwd is None:
        # Fall back to the repo path (dev mode without ~/.sutando/workspace)
        cwd = Path(__file__).resolve().parent.parent.parent.parent

    def _run():
        try:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
            return r.stdout.strip() or r.stderr.strip()
        except Exception as e:
            return f"summon failed: {e}"

    out = await asyncio.to_thread(_run)
    await params.result_callback({"status": "done", "result": out or "Zoom is up."})


# ===========================================================================
# registration helpers
# ===========================================================================


ALL_SCHEMAS: list[FunctionSchema] = [
    WORK_SCHEMA,
    CURRENT_TIME_SCHEMA,
    RECENT_CONTEXT_SCHEMA,
    SAVE_NOTE_SCHEMA,
    READ_NOTE_SCHEMA,
    SUMMON_SCHEMA,
]

_HANDLERS: dict[str, Any] = {
    "work": work_handler,
    "get_current_time": current_time_handler,
    "recent_context": recent_context_handler,
    "save_note": save_note_handler,
    "read_note": read_note_handler,
    "summon": summon_handler,
}


def register_all(llm: LLMService) -> ToolsSchema:
    """Register every tool with the LLM service and return the ToolsSchema.

    Caller passes the returned ToolsSchema into the LLMContext so the model
    sees the function definitions on every request.
    """
    for schema in ALL_SCHEMAS:
        # `work` is a long-running task — set a generous timeout. Others
        # are quick.
        timeout_secs = 365 * 60 if schema.name == "work" else 60
        # cancel_on_interruption=False so a barge-in doesn't abandon a
        # work-tool that's already queued a task file; the result will
        # still arrive even if the user interrupted the spoken status.
        llm.register_function(
            schema.name,
            _HANDLERS[schema.name],
            cancel_on_interruption=False,
            timeout_secs=timeout_secs,
        )
    return ToolsSchema(standard_tools=ALL_SCHEMAS)
