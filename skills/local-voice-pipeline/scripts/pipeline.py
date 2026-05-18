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
import os
import sys
import urllib.request
import warnings
from pathlib import Path

# faster-whisper emits divide/overflow/invalid RuntimeWarnings from its mel-spectrogram
# matmul whenever it processes a silent or near-silent audio chunk. Cosmetic, harmless.
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
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
KOKORO_VOICE_ID = "af_heart"  # warm female; swap to "am_michael" for male

SYSTEM_PROMPT = (
    "You are Sutando, the user's personal AI agent — friendly, capable, and "
    "succinct. Speak naturally, as if talking out loud. Replies are usually "
    "two or three sentences (sometimes one for yes/no). Never reply with a "
    "single word. No markdown, no lists.\n\n"
    "TOOL USE: For any non-trivial request — email, calendar, calls, "
    "research, writing, code, file operations, web lookups, scheduling — "
    "call the `work` tool. Pass the user's request verbatim plus any "
    "context that helps the executor. While `work` runs (it may take "
    "seconds to minutes), tell the user you're on it; never claim it's "
    "done. The result will come back to you and you'll relay it.\n\n"
    "For quick built-ins (the current time, your recent context, saving "
    "or reading a note, summoning a Zoom call) use the matching tool "
    "directly instead of `work`. Don't speculate when a tool can answer."
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


def build_audio_filter():
    """Optional input-side audio filter (echo cancel / noise suppression).

    None of these are wired by default — they need extra installs or an API
    key. Trade-offs:

    * ``KRISP_API_KEY``: best-quality AEC + denoise, lets you ditch
      headphones and re-enable barge-in. Needs ``pip install
      pipecat-ai[krisp]`` + a Krisp account (free 200 hr/mo tier).
    * ``RNNOISE_ENABLED=1``: noise suppression only (not echo cancel), free.
      Helps with fan/hum/keyboard, not with mic-hears-speakers. Needs
      ``pip install pipecat-ai[rnnoise]``.
    * default: no filter. Headphones recommended for clean barge-in.
    """
    if os.environ.get("KRISP_API_KEY"):
        try:
            from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
            logger.info("audio filter: Krisp (AEC + denoise)")
            return KrispVivaFilter()
        except ImportError as e:
            logger.warning(f"KRISP_API_KEY set but krisp_audio missing: {e}")
            logger.warning("install with: pip install pipecat-ai[krisp]")
    if os.environ.get("RNNOISE_ENABLED") == "1":
        try:
            from pipecat.audio.filters.rnnoise_filter import RNNoiseFilter
            logger.info("audio filter: RNNoise (denoise only; no echo cancel)")
            return RNNoiseFilter()
        except ImportError as e:
            logger.warning(f"RNNOISE_ENABLED=1 but module missing: {e}")
            logger.warning("install with: pip install pipecat-ai[rnnoise]")
    logger.info("audio filter: none (use headphones for clean barge-in)")
    return None


async def main() -> None:
    ensure_kokoro_assets()

    audio_filter = build_audio_filter()

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            audio_in_filter=audio_filter,
        )
    )

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

    # Qwen3 reasoning OFF — `chat_template_kwargs.enable_thinking: false` goes
    # in the request body via OpenAILLMSettings.extra. Without this Qwen3 burns
    # 80%+ of completion tokens in `reasoning_content` and voice TTFB suffers.
    llm = OpenAILLMService(
        model=LLAMA_MODEL,
        base_url=LLAMA_URL,
        api_key="local",
    )
    # Pipecat's OpenAILLMSettings.extra is spread directly into the OpenAI SDK
    # call as kwargs (see base_llm.build_chat_completion_params: params.update(extra)).
    # So the right key is "extra_body" — the SDK then forwards its contents to the
    # server's request body, which is where llama.cpp picks up chat_template_kwargs.
    llm._settings.extra = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }

    # Register Sutando tools (work + native helpers). Returns the
    # ToolsSchema that the LLMContext needs so the model sees these
    # functions on every request.
    tools_schema = register_tools(llm)

    tts = KokoroTTSService(
        voice_id=KOKORO_VOICE_ID,
        model_path=str(KOKORO_MODEL),
        voices_path=str(KOKORO_VOICES),
    )

    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=tools_schema,
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            # NB: `allow_interruptions` doesn't exist on PipelineParams in 1.2.x.
            # Interruption control is wired through user_turn_strategies above.
            enable_metrics=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("🎙  mic is hot — speak when ready")

    runner = PipelineRunner()

    logger.info("=" * 60)
    logger.info("SUTANDO LOCAL VOICE PIPELINE — Phase 1 PoC")
    logger.info(f"  LLM:    {LLAMA_MODEL} @ {LLAMA_URL}")
    logger.info(f"  STT:    Whisper base (Metal)")
    logger.info(f"  TTS:    Kokoro ({KOKORO_VOICE_ID})")
    logger.info(f"  VAD:    Silero (stop_secs=0.3, snappier)")
    logger.info("  Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        await runner.run(task)
    except KeyboardInterrupt:
        logger.info("stopping...")


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
