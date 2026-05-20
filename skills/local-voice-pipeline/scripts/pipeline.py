#!/usr/bin/env python3
"""Realtime local voice pipeline — Pipecat graph wired to the turboquant llama-server.

Mic → Silero VAD → faster-whisper STT → llama-server (Qwen3, reasoning off)
→ Kokoro TTS → speaker.

Prereqs:
    - llama-server running at http://127.0.0.1:8081 (see ./llama-server.sh)
    - venv activated: source .venv/bin/activate
    - Kokoro model: auto-downloaded on first run if not present in models/

Usage:
    ./pipeline.py

Then speak into your mic. Ctrl+C to stop. Conversation history is held in the
LLMContext for the duration of the session — no persistent memory yet.

Phase 1 scope: realtime conversational loop, no tool calling. Phase 2 will
port Sutando's ~30 inline tools into Pipecat function definitions.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import os
import sys
import urllib.request


# Load env from common Sutando + Hermes locations BEFORE anything reads
# os.environ. Hermes (~/.hermes/.env) holds shared cloud creds the rest of
# the fleet uses — including AZURE_FOUNDRY_SWEDEN_KEY for Haiku 4.5. Order:
# repo .env first (project overrides), then ~/.hermes/.env (shared).
def _load_env_files() -> None:
    from pathlib import Path as _P
    for envfile in (
        _P(__file__).resolve().parents[3] / ".env",  # sutando repo .env
        _P.home() / ".hermes" / ".env",              # shared hermes .env
    ):
        if not envfile.exists():
            continue
        try:
            for line in envfile.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            pass


_load_env_files()
import warnings
from datetime import datetime, timezone
from pathlib import Path

# faster-whisper emits divide/overflow/invalid RuntimeWarnings from its mel-spectrogram
# matmul whenever it processes a silent or near-silent audio chunk. Cosmetic, harmless.
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from tools import register_all as register_tools

# --- config ---

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

LLAMA_URL = "http://127.0.0.1:8081/v1"
LLAMA_MODEL = "qwen3"

# Kokoro: small ONNX model + voices binary. Auto-download on first run.
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_MODEL = MODELS / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS / "voices-v1.0.bin"
_DEFAULT_KOKORO_VOICE_ID = "bf_alice"  # British female; full list in tools.list_voices


def _load_runtime_config() -> dict:
    """Read state/pipeline-config.json — voice + audio device overrides
    written by the `set_voice` / `set_audio_device` tools. Missing file or
    invalid JSON falls back to defaults silently (first-run case)."""
    cfg_path = Path(
        os.environ.get("SUTANDO_WORKSPACE") or Path.home() / ".sutando" / "workspace"
    ).expanduser() / "state" / "pipeline-config.json"
    try:
        if cfg_path.exists():
            return json.loads(cfg_path.read_text())
    except Exception:
        pass
    return {}


_RUNTIME_CFG = _load_runtime_config()
KOKORO_VOICE_ID = _RUNTIME_CFG.get("voice_id") or _DEFAULT_KOKORO_VOICE_ID
INPUT_DEVICE_INDEX = _RUNTIME_CFG.get("input_device_index")  # None = system default
OUTPUT_DEVICE_INDEX = _RUNTIME_CFG.get("output_device_index")  # None = system default

# UI hook: write per-turn transcripts to state/voice-session.jsonl and a
# liveness/state snapshot to state/voice-status.json. Workspace resolution
# mirrors `src/workspace_default.py` (env override → ~/.sutando/workspace).
_WORKSPACE_DIR = Path(
    os.environ.get("SUTANDO_WORKSPACE")
    or Path.home() / ".sutando" / "workspace"
).expanduser()
_STATE_DIR = _WORKSPACE_DIR / "state"
_SESSION_LOG = _STATE_DIR / "voice-session.jsonl"
_STATUS_FILE = _STATE_DIR / "voice-status.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(state: str, **extra: object) -> None:
    """Best-effort write of the current pipeline state. Never raises."""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"backend": "pipecat", "state": state, "ts": _now_iso()}
        payload.update(extra)
        _STATUS_FILE.write_text(json.dumps(payload))
    except Exception:
        pass


def _log_turn(role: str, text: str) -> None:
    """Best-effort JSONL append of a single turn. Never raises."""
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        with _SESSION_LOG.open("a") as f:
            f.write(json.dumps({"ts": _now_iso(), "role": role, "text": text}) + "\n")
    except Exception:
        pass


_LAST_STATE: dict[str, object] = {"state": "starting"}


def _write_status_track(state: str, **extra: object) -> None:
    """Write status AND remember the last state for heartbeat refreshes."""
    _LAST_STATE["state"] = state
    _LAST_STATE.update(extra)
    _write_status(state, **extra)


async def _heartbeat_task(interval: float = 30.0) -> None:
    """Re-write the status file every ~30s with the last known state.

    The dashboard treats the status file as stale after 120s of mtime
    silence and hides the panel — which means a healthy-but-idle pipeline
    (no one's spoken in a few minutes) looks dead in the UI. This keeps
    mtime fresh without altering the actual state.
    """
    while True:
        try:
            await asyncio.sleep(interval)
            state = str(_LAST_STATE.get("state", "listening"))
            extra = {k: v for k, v in _LAST_STATE.items() if k != "state"}
            _write_status(state, **extra)
        except asyncio.CancelledError:
            break
        except Exception:
            pass


class TranscriptTap(BaseObserver):
    """Observes every frame pushed across the pipeline.

    Attached to PipelineTask via `observers=[...]`. Unlike a FrameProcessor,
    an observer sits OUTSIDE the linear pipeline so frames consumed by
    intermediate processors (e.g. context_aggregator consumes TranscriptionFrame
    before it reaches the end of the pipeline) are still visible here.

    Dedup happens at the (role, text) layer — Pipecat creates separate
    TTSTextFrame instances for the same text at multiple graph hops
    (assistant_aggregator output → TTS service input), so `frame.id`
    dedup leaks duplicates with different ids but identical content
    (observed 2026-05-19: "However, I can adjust my tone…" logged at
    09:15:11 AND 09:15:17 from one Kokoro `Generating TTS`). A short
    text-keyed buffer + recency check catches both copies cheaply.
    """

    def __init__(self) -> None:
        super().__init__()
        # Bounded ring of recently-emitted (role, text) keys. 64 is plenty
        # for typical 3-5-sentence assistant turns; older entries fall off
        # as new lines come in, so distant repeats still log (intentional —
        # the user repeating themselves is a legitimate transcript event).
        self._recent: list[tuple[str, str]] = []

    def _is_duplicate(self, role: str, text: str) -> bool:
        key = (role, text.strip())
        if key in self._recent:
            return True
        self._recent.append(key)
        if len(self._recent) > 64:
            self._recent.pop(0)
        return False

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        try:
            if isinstance(frame, TranscriptionFrame) and getattr(frame, "text", ""):
                if self._is_duplicate("user", frame.text):
                    return
                _log_turn("user", frame.text)
                _write_status_track("thinking")
            elif isinstance(frame, TTSTextFrame) and getattr(frame, "text", ""):
                # TTSTextFrame is the sentence-level aggregate that gets fed
                # into the TTS service. One frame per spoken sentence — much
                # cleaner than re-assembling per-token LLMTextFrames.
                if self._is_duplicate("assistant", frame.text):
                    return
                _log_turn("assistant", frame.text)
            elif isinstance(frame, TTSStartedFrame):
                _write_status_track("speaking")
            elif isinstance(frame, TTSStoppedFrame):
                _write_status_track("listening")
            elif isinstance(frame, UserStartedSpeakingFrame):
                _write_status_track("user_speaking")
            elif isinstance(frame, UserStoppedSpeakingFrame):
                _write_status_track("transcribing")
        except Exception:
            pass


def _load_stand_identity() -> str:
    """Mirror voice-agent.ts: read stand-identity.json from workspace; return
    a sentence describing the stand name or empty string on miss."""
    try:
        for p in (_WORKSPACE_DIR / "stand-identity.json",
                  Path(__file__).resolve().parents[2] / "stand-identity.json"):
            if p.exists():
                si = json.loads(p.read_text())
                if si.get("name"):
                    origin = si.get("nameOrigin", "earned through use")
                    return (
                        f"Your Stand name is {si['name']}. Origin: {origin}. "
                        f"When asked your name or who you are, say "
                        f"\"I'm Sutando — {si['name']}.\""
                    )
    except Exception:
        pass
    return ""


def _load_voice_context() -> str:
    """Mirror voice-agent.ts: try $SUTANDO_PRIVATE_DIR/voice-contexts/<active>.txt,
    then repo voice-context.txt. Silent fallback on miss."""
    try:
        private_root = os.environ.get("SUTANDO_PRIVATE_DIR")
        if private_root:
            root = Path(private_root).expanduser()
            pointer = root / "voice-contexts" / "active"
            if pointer.exists():
                name = pointer.read_text().strip()
                if name and re.fullmatch(r"[A-Za-z0-9._-]+", name):
                    ctx = root / "voice-contexts" / f"{name}.txt"
                    if ctx.exists():
                        return ctx.read_text()
    except Exception:
        pass
    try:
        fallback = Path(__file__).resolve().parents[2] / "voice-context.txt"
        if fallback.exists():
            return fallback.read_text()
    except Exception:
        pass
    return ""


def _build_system_prompt(tools_enabled: bool) -> str:
    """Ported from src/voice-agent.ts `session.instructions`. Static parts
    only. Tool-routing guidance branches on whether the LVP_ENABLE_TOOLS
    code path actually registered functions: with tools the model should
    CALL `work` (and let the main Sutando act); without tools it should
    verbally relay and be honest about its lack of agency."""
    parts = [
        "You are Sutando, a personal AI that belongs entirely to the user.",
        "Named after Stands from JoJo's Bizarre Adventure — a personal spirit that fights for you.",
        "Every Sutando evolves differently based on what its user needs.",
        _load_stand_identity(),
        _load_voice_context(),
        "You handle anything: research, writing, email, scheduling, code, "
        "logistics, phone calls, meetings, creative work.",
        "You can join Google Meet and Zoom meetings, make phone calls, see "
        "the user's screen, and reach them on Telegram, Discord, web, or phone.",
        "You build a model of the user over time — their preferences, "
        "working style, voice, and priorities — and use it without them "
        "having to repeat themselves.",
        "",
    ]
    if tools_enabled:
        parts += [
            # Mirrors the Gemini voice-agent's "DEFAULT BEHAVIOR: call work"
            # rule. Without this the model verbally relays and never fires
            # the function call.
            "DEFAULT BEHAVIOR: call `work` for almost everything. You are "
            "the voice interface; the main Sutando is the brain. Your job "
            "is to relay the user's request to `work` and speak the result.",
            "",
            "Call `work` for: checking email, calendar, sending messages, "
            "running code, web search beyond a quick fact, opening apps, "
            "joining meetings, making calls, anything that touches the "
            "user's system or external services, anything that needs "
            "research or multi-step reasoning.",
            "",
            "Answer directly (no `work`) only for: simple greetings, "
            "self-introduction, yes/no acknowledgments, asking a clarifying "
            "question, get_current_time, recent_context, save_note, "
            "read_note, summon — those are instant native tools, call them "
            "directly when relevant.",
            "",
            "When you call `work`, briefly tell the user what you're doing "
            "(\"checking your email…\") so they know the wait is intentional. "
            "When the result comes back, speak it naturally — don't read it "
            "verbatim if it's long, summarize.",
            "",
            # Anti-hallucination guard. Qwen3-8B will otherwise say
            # "let me check…", "I'll make sure…", "one moment please"
            # without ever firing a tool. The phrase-level bans below force
            # tool-first, narration-second.
            "CRITICAL — TOOL FIRST, NARRATION SECOND:",
            "- BEFORE saying any of these phrases, you MUST have already "
            "fired a tool call in this turn: \"let me check\", \"let me "
            "look\", \"let me see\", \"let me adjust\", \"let me update\", "
            "\"I'll make sure\", \"I'll set\", \"I'll handle\", \"I'll fix\", "
            "\"I'll configure\", \"I'll update\", \"I'll change\", \"one "
            "moment\", \"give me a moment\", \"hold on\", \"working on it\". "
            "If you have NOT called a tool yet, do NOT say those words. The "
            "tool call comes first; the narration describes what happened.",
            "",
            "- If you find yourself wanting to commit to an action, the "
            "correct sequence is: (1) call `work` (or a native tool) with "
            "the task; (2) when the result comes back, speak about what "
            "actually happened. NOT: speak first, claim you'll do it, hope "
            "it works out.",
            "",
            "- The native tools available are: set_voice (Kokoro voice), "
            "list_audio_devices (mic/speakers), set_audio_device (switch "
            "mic or speakers), save_note, read_note, get_current_time, "
            "recent_context, summon (Zoom). For ANYTHING ELSE that touches "
            "the user's system, files, services, or external world: call "
            "`work`. There is no third option.",
            "",
            "- If you can't do something at all — say so PLAINLY in one "
            "sentence and stop. Don't pad with \"but I'll check anyway\" "
            "or \"let me see if there's a way\". Those are lies.",
            "",
            "- You CANNOT remember preferences across sessions on your own. "
            "If the user says \"remember this\", call `save_note` (or `work` "
            "with a save-to-memory task) — don't just say \"got it\".",
        ]
    else:
        parts += [
            "IMPORTANT — local-voice mode (no tools): right now you are "
            "running on a local pipeline that does NOT have tool access. "
            "You cannot actually send email, check calendar, run code, or "
            "act on the user's system from this voice channel. When the "
            "user asks you to DO something (\"check my email\", \"send X\", "
            "\"open Y\"), acknowledge briefly and say you'll relay it to "
            "the main Sutando — do NOT pretend you did it. For questions, "
            "conversation, advice, brainstorming, explanations — answer "
            "fully and naturally.",
        ]
    parts += [
        "",
        # Voice-style guardrails from the Phase 1 prompt.
        "Voice style: give complete, natural answers — usually two or three "
        "sentences, sometimes one if the question is yes/no. Never reply "
        "with just a single word. No markdown, no lists. Speak naturally "
        "as if talking out loud. If you don't know something, say so briefly.",
    ]
    return "\n".join(p for p in parts if p is not None)


SYSTEM_PROMPT = _build_system_prompt(
    tools_enabled=os.environ.get("LVP_ENABLE_TOOLS") == "1"
)


def ensure_kokoro_assets() -> None:
    """Download Kokoro model + voices on first run."""
    MODELS.mkdir(exist_ok=True)
    if not KOKORO_MODEL.exists():
        logger.info(f"downloading {KOKORO_MODEL.name}...")
        urllib.request.urlretrieve(KOKORO_MODEL_URL, KOKORO_MODEL)
    if not KOKORO_VOICES.exists():
        logger.info(f"downloading {KOKORO_VOICES.name}...")
        urllib.request.urlretrieve(KOKORO_VOICES_URL, KOKORO_VOICES)
    logger.info(f"kokoro assets ready ({KOKORO_MODEL.stat().st_size//1_000_000} MB model + "
                f"{KOKORO_VOICES.stat().st_size//1_000_000} MB voices)")


_MUTE_SENTINEL = _STATE_DIR / "pipeline-muted.sentinel"


from pipecat.audio.filters.base_audio_filter import BaseAudioFilter as _BaseAudioFilter  # noqa: E402


class _MuteSentinelFilter(_BaseAudioFilter):
    """Drops mic audio (replaces with silence) when the mute sentinel file
    exists. Composes with an optional inner filter so noise-suppression /
    AEC still runs when the user is unmuted.

    Implements the BaseAudioFilter contract (start/stop/process_frame/filter)
    by delegating to the wrapped filter when present.

    Sentinel check is mtime-cached: stat() is microseconds but we still
    cache to keep the hot audio path tight (~50 calls/sec at 20ms chunks).
    """

    def __init__(self, inner=None) -> None:
        self._inner = inner
        self._last_check_ts = 0.0
        self._muted = False

    def _is_muted(self) -> bool:
        # Re-check at most every 100ms — well under any human's reaction time
        # to a mute toggle, but cheap enough that audio frames don't stat-storm.
        now = time.time()
        if now - self._last_check_ts > 0.1:
            self._muted = _MUTE_SENTINEL.exists()
            self._last_check_ts = now
        return self._muted

    async def start(self, sample_rate: int):
        if self._inner is not None:
            await self._inner.start(sample_rate)

    async def stop(self):
        if self._inner is not None:
            await self._inner.stop()

    async def process_frame(self, frame):
        if self._inner is not None:
            await self._inner.process_frame(frame)

    async def filter(self, audio: bytes) -> bytes:
        if self._is_muted():
            # Return same-length silence so downstream timing stays correct
            # (VAD won't trigger on zeros — exactly the goal).
            return b"\x00" * len(audio)
        if self._inner is not None:
            return await self._inner.filter(audio)
        return audio


def build_audio_filter():
    """Input-side audio filter chain. Always wraps the chosen inner filter
    (or None) in a MuteSentinelFilter so the web UI can mute/unmute by
    creating/removing state/pipeline-muted.sentinel.

    Inner-filter choices (none wired by default — extra installs / API key):

    * ``KRISP_API_KEY``: best-quality AEC + denoise, lets you ditch
      headphones and re-enable barge-in. Needs ``pip install
      pipecat-ai[krisp]`` + a Krisp account (free 200 hr/mo tier).
    * ``RNNOISE_ENABLED=1``: noise suppression only (not echo cancel), free.
      Helps with fan/hum/keyboard, not with mic-hears-speakers. Needs
      ``pip install pipecat-ai[rnnoise]``.
    * default: no inner filter. Headphones recommended for clean barge-in.
    """
    inner = None
    if os.environ.get("KRISP_API_KEY"):
        try:
            from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
            logger.info("audio filter: Krisp (AEC + denoise) + mute-sentinel")
            inner = KrispVivaFilter()
        except ImportError as e:
            logger.warning(f"KRISP_API_KEY set but krisp_audio missing: {e}")
            logger.warning("install with: pip install pipecat-ai[krisp]")
    if inner is None and os.environ.get("RNNOISE_ENABLED") == "1":
        try:
            from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
            logger.info("audio filter: RNNoise (denoise only; no echo cancel) + mute-sentinel")
            inner = RNNoiseFilter()
        except ImportError as e:
            logger.warning(f"RNNOISE_ENABLED=1 but module missing: {e}")
            logger.warning("install with: pip install pipecat-ai[rnnoise]")
    if inner is None:
        logger.info("audio filter: mute-sentinel only (no inner; use headphones for clean barge-in)")
    return _MuteSentinelFilter(inner=inner)


async def main() -> None:
    ensure_kokoro_assets()

    audio_filter = build_audio_filter()

    # Build params dict so we only pass device indices when set (None vs
    # missing kwarg differs in some pipecat adapters; safer to omit).
    transport_kwargs = dict(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=48000,
        audio_in_filter=audio_filter,
    )
    if INPUT_DEVICE_INDEX is not None:
        transport_kwargs["input_device_index"] = INPUT_DEVICE_INDEX
        logger.info(f"audio input: device index {INPUT_DEVICE_INDEX} (override)")
    if OUTPUT_DEVICE_INDEX is not None:
        transport_kwargs["output_device_index"] = OUTPUT_DEVICE_INDEX
        logger.info(f"audio output: device index {OUTPUT_DEVICE_INDEX} (override)")
    transport = LocalAudioTransport(params=LocalAudioTransportParams(**transport_kwargs))

    # VAD is a separate FrameProcessor in Pipecat 1.2.x — NOT a transport param.
    # Sits between transport.input() and stt in the pipeline graph.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.3,    # snappier turn-end (was 0.5; the 200ms diff is felt)
                confidence=0.6,
            )
        ),
    )

    # int8 is ~25% faster than the float16 default on Metal with no accuracy
    # loss at base size. Falls back to default if CTranslate2 wheel lacks
    # int8 (rare on macOS).
    try:
        stt = WhisperSTTService(
            model="base",
            device="auto",        # Metal on M-series
            compute_type="int8",  # see above
        )
    except Exception as e:
        logger.warning(f"int8 Whisper failed ({e}); falling back to default compute_type")
        stt = WhisperSTTService(model="base", device="auto", compute_type="default")

    # LLM backend: Azure AI Foundry → Claude Haiku 4.5 (preferred — better
    # instruction-following than local Qwen3-8B, way fewer hallucinated
    # "I'll do X" commitments). Falls back to local llama-server if
    # AZURE_FOUNDRY_SWEDEN_KEY isn't set (e.g. fresh checkout / dev).
    foundry_key = os.environ.get("AZURE_FOUNDRY_SWEDEN_KEY")
    foundry_url = os.environ.get("AZURE_FOUNDRY_SWEDEN_BASE_URL")
    if foundry_key and foundry_url:
        from pipecat.services.anthropic.llm import AnthropicLLMService
        from anthropic import Anthropic
        # Anthropic SDK with a custom base_url routes to Azure AI Foundry's
        # native Anthropic-compat endpoint (https://<resource>.services.ai
        # .azure.com/anthropic). Authentication is the standard x-api-key
        # header, which the SDK populates from api_key.
        anth_client = Anthropic(api_key=foundry_key, base_url=foundry_url)
        llm = AnthropicLLMService(
            api_key=foundry_key,  # required positionally even when client is passed
            model="claude-haiku-4-5",
            client=anth_client,
        )
        logger.info(f"LLM: Claude Haiku 4.5 via Azure AI Foundry ({foundry_url})")
    else:
        # Local fallback: Qwen3 via llama-server. Reasoning OFF —
        # `chat_template_kwargs.enable_thinking: false` goes in the request
        # body via OpenAILLMSettings.extra. Without this Qwen3 burns 80%+
        # of completion tokens in `reasoning_content` and voice TTFB suffers.
        llm = OpenAILLMService(
            model=LLAMA_MODEL,
            base_url=LLAMA_URL,
            api_key="local",
        )
        llm._settings.extra = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }
        logger.info(f"LLM: Qwen3-8B via local llama-server ({LLAMA_URL})")

    # Register Sutando tools (work + native helpers). Returns the
    # ToolsSchema that the LLMContext needs so the model sees these
    # functions on every request.
    # NOTE 2026-05-18 — temporarily disabled. Phase 2 tool registration is
    # breaking the LLM→TTS path: llama-server returns 200 but no Pipecat
    # `Generating TTS` log appears, and no audio is transmitted. Likely
    # Pipecat is consuming tool-call frames silently instead of speaking.
    # Toggle back on once we've reproduced and pinpointed the issue.
    tools_schema = None
    if os.environ.get("LVP_ENABLE_TOOLS") == "1":
        tools_schema = register_tools(llm)

    tts = KokoroTTSService(
        voice_id=KOKORO_VOICE_ID,
        model_path=str(KOKORO_MODEL),
        voices_path=str(KOKORO_VOICES),
    )

    # Only pass tools= when actually registered (None vs missing kwarg
    # behave differently in some Pipecat adapters; safer to omit).
    if tools_schema is not None:
        context = LLMContext(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}],
            tools=tools_schema,
        )
    else:
        context = LLMContext(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        )

    # Interruption control in Pipecat 1.2.x lives on the turn-start strategies,
    # NOT on PipelineParams (`allow_interruptions` was silently dropped from
    # PipelineParams by Pydantic). To stop the agent from cutting itself off
    # when the mic hears the speaker, build the strategies with interruptions
    # disabled.
    user_turn_strategies = UserTurnStrategies(
        start=[
            VADUserTurnStartStrategy(enable_interruptions=False),
            TranscriptionUserTurnStartStrategy(enable_interruptions=False),
        ]
    )
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=user_turn_strategies,
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        vad,                            # detect speech start/stop (Silero)
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    # TranscriptTap rides as a pipeline observer (not a processor) — it sees
    # every frame pushed between every pair of processors, including
    # TranscriptionFrame frames that `context_aggregator.user()` consumes
    # before they'd reach a downstream tap. Writes voice-session.jsonl +
    # voice-status.json so the dashboard at :7844 can show live state.
    transcript_tap = TranscriptTap()

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            # NB: `allow_interruptions` doesn't exist on PipelineParams in 1.2.x.
            # Interruption control is wired through user_turn_strategies above.
            enable_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
        observers=[transcript_tap],
        # Pipecat's default cancels the pipeline after 5 min with no
        # bot/user-speaking frames. Sutando is a persistent voice agent
        # that should sit silently for hours waiting for the user, so we
        # disable the idle cancel rather than nudge the timeout up.
        cancel_on_idle_timeout=False,
    )

    # NOTE: LocalAudioTransport doesn't emit `on_client_connected` — pipecat
    # logs a warning if we register it. The mic opens as soon as the pipeline
    # is running, so we flip the status to "listening" right before runner.run().
    runner = PipelineRunner()
    _write_status_track("starting")
    logger.info("🎙  mic is hot — speak when ready")
    _write_status_track("listening")

    logger.info("=" * 60)
    logger.info("SUTANDO LOCAL VOICE PIPELINE — Phase 1 PoC")
    if os.environ.get("AZURE_FOUNDRY_SWEDEN_KEY"):
        logger.info(f"  LLM:    Haiku 4.5 @ Azure AI Foundry (Sweden)")
    else:
        logger.info(f"  LLM:    {LLAMA_MODEL} @ {LLAMA_URL}")
    logger.info(f"  STT:    Whisper base (Metal)")
    logger.info(f"  TTS:    Kokoro ({KOKORO_VOICE_ID})")
    logger.info(f"  VAD:    Silero (stop_secs=0.3, snappier)")
    logger.info("  Ctrl+C to stop")
    logger.info("=" * 60)

    heartbeat = asyncio.create_task(_heartbeat_task())
    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("stopping...")
    finally:
        heartbeat.cancel()
        _write_status_track("stopped")


if __name__ == "__main__":
    # Sanity check: llama-server must be up.
    try:
        with urllib.request.urlopen(LLAMA_URL.replace("/v1", "/health"), timeout=2) as r:
            if r.status != 200:
                logger.error("llama-server not healthy at " + LLAMA_URL)
                sys.exit(2)
    except Exception as e:
        logger.error(f"llama-server not reachable at {LLAMA_URL}: {e}")
        logger.error("start it first: ./scripts/llama-server.sh")
        sys.exit(2)

    asyncio.run(main())
