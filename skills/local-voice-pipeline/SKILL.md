---
name: local-voice-pipeline
description: Pipecat-based local realtime voice pipeline (Whisper STT → llama.cpp Qwen3-8B → Kokoro TTS). Alternative to voice-agent.ts's Gemini Live for fully local operation. Phase 1+2 shipped.
trigger: /local-voice-pipeline
---

# Local Voice Pipeline

A from-scratch realtime voice runtime that doesn't depend on Gemini Live or any cloud LLM. Built 2026-05-18 to replace the bodhi/Gemini Live path after the monthly cap exceeded incident.

## Stack (M-series Mac, what's actually wired)

| Component | Choice | Notes |
|---|---|---|
| Orchestrator | Pipecat 1.2.1 (Python) | Graph: mic → VAD → STT → ctx → LLM → TTS → speaker |
| VAD | Silero v5 via `VADProcessor` | `stop_secs=0.3` for snappy turn-end |
| STT | faster-whisper `base`, `compute_type=int8` | Metal on M-series, ~25% faster than float16 |
| LLM | llama.cpp turboquant fork + Qwen3-8B-UD-Q5_K_XL.gguf | OpenAI-compat at :8081; reasoning OFF |
| TTS | Kokoro `af_heart` (ONNX, ~80MB model + 5MB voices) | Auto-downloads on first run |
| Tool surface | `scripts/tools.py` — `work` + 5 native helpers | `work` fans out to core via tasks/ bridge |

## How it differs from bodhi

bodhi (voice-agent.ts) lives in TypeScript and talks Gemini Live over WebSocket. It loads Sutando's ~30 inline tools directly. Latency depends on cloud round-trip; monthly cost depends on Gemini usage.

This pipeline lives in Python (under `.venv/`) and runs entirely on-device. It exposes only **one big tool — `work` —** plus a handful of low-latency native helpers (`get_current_time`, `recent_context`, `save_note`, `read_note`, `summon`). The `work` tool writes the same task-file format that bodhi writes, so the core (Claude Code session) handles the full 30-tool surface transparently. Net effect: pipecat path keeps full Sutando capability with one Python function, no tool-by-tool port required.

## Switching backends

Set in `.env`:
```
VOICE_BACKEND=pipecat   # local stack (this skill)
VOICE_BACKEND=bodhi     # original Gemini Live path (default if unset)
```

`bash src/startup.sh` reads this and starts the right stack. For `pipecat`, it starts llama-server in the background but leaves `pipeline.py` for you to launch in an interactive terminal (it opens the local mic):
```
cd skills/local-voice-pipeline
source .venv/bin/activate
python scripts/pipeline.py
```

## Audio filter (echo cancel / denoise)

By default no input filter is wired. Mic-hears-speakers feedback is the main symptom — it causes the agent to interrupt itself if barge-in is enabled. Three options:

| Env | Effect | Cost |
|---|---|---|
| (unset) | No filter. Use headphones. | $0 |
| `RNNOISE_ENABLED=1` | RNNoise denoise only (NOT echo cancel) — helps fan/hum, not feedback | $0, `pip install pipecat-ai[rnnoise]` |
| `KRISP_API_KEY=...` | Full Krisp AEC + denoise — go hands-free, re-enable barge-in | Free 200hr/mo, `pip install pipecat-ai[krisp]` |

Barge-in is currently disabled in `pipeline.py` (`enable_interruptions=False` on turn-start strategies) — mic-echo cuts TTS off after one word otherwise. Turn it back on when Krisp is wired.

## Latency knobs already tuned

- Whisper `compute_type=int8` (~25% faster STT than float16)
- VAD `stop_secs=0.3` (~200ms snappier turn-end)
- Qwen3 reasoning OFF via `extra_body.chat_template_kwargs.enable_thinking: false` (132→13 completion tokens on yes/no answers; 5.5s→3.0s LLM TTFB)

Further knobs available without code: `KOKORO_VOICE_ID` env, swap Whisper to `large-v3-turbo` for better accuracy at the cost of latency.

## What's NOT wired yet (Phase 3+)

- **Barge-in** — works in principle but needs AEC (see above).
- **Recording / vision tools** — Sutando's `recording-tools.ts` and `vision-tools.ts` aren't in Pipecat's tool registry. They're reachable via `work` (core can call them), but with one extra hop of latency.
- **Persistent memory** — conversation context resets on each `pipeline.py` invocation. Across-restart memory would need to hook into Sutando's existing memory directory.
- **Phone path** — the phone-conversation skill still uses bodhi. Phase 4.

## Files (under `skills/local-voice-pipeline/`)

```
SKILL.md                      this file
scripts/install.sh            one-shot setup (brew + uv venv + pip)
scripts/smoke-test.py         9-check import test
scripts/llama-server.sh       starts turboquant llama-server on :8081
scripts/demo-roundtrip.py     text→LLM→TTS hello-world (no mic)
scripts/pipeline.py           full Pipecat graph (mic→...→speaker)
scripts/tools.py              Pipecat function defs (work + 5 native helpers)
models/Qwen3-8B-…gguf         symlink to ~/src/llama-cpp-turboquant/models/
models/kokoro-v1.0.onnx       auto-downloaded by pipeline.py
models/voices-v1.0.bin        auto-downloaded by pipeline.py
logs/llama-server.log         (created at runtime)
```

## Pipecat 1.2.x foot-guns (silently-dropped params)

In order of how-much-time-they-cost-me:

1. `LocalAudioTransportParams.vad_analyzer` doesn't exist — VAD is a separate `VADProcessor` in the pipeline graph.
2. `PipelineParams.allow_interruptions` doesn't exist — wire it through `UserTurnStrategies(start=[VADUserTurnStartStrategy(enable_interruptions=False), ...])`.
3. `OpenAILLMSettings.extra` is spread as **kwargs**, not into the request body — to pass server flags wrap them in `extra_body`.

All three are silently dropped by Pydantic's `extra=ignore` config. There's no warning. If a config option looks like it should exist and does nothing, check that it's actually a field on the model.
