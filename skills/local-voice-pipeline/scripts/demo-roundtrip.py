#!/usr/bin/env python3
"""End-to-end demo without mic: text → LLM (turboquant llama-server) → TTS (edge-tts) → audio file.

Validates the LLM+TTS legs of the Pipecat stack work cleanly. STT integration
comes when pipeline.py adds the mic path.

Usage:
    ./demo-roundtrip.py "What's the best thing about realtime voice agents?"
"""
import asyncio
import sys
import time
from pathlib import Path

import edge_tts
from openai import OpenAI

LLAMA_URL = "http://127.0.0.1:8081/v1"
LLAMA_MODEL = "qwen3"
TTS_VOICE = "en-GB-RyanNeural"  # change for different timbre

OUT_DIR = Path(__file__).resolve().parent.parent / "logs"
OUT_DIR.mkdir(exist_ok=True)


async def main(prompt: str) -> int:
    print(f"\n>>> prompt: {prompt}\n")

    # --- LLM leg ---
    # Qwen3 ships in "thinking" mode by default — reasoning_content burns 80%+
    # of completion tokens before any user-visible text. For voice latency we
    # disable thinking via Qwen3's chat_template_kwargs flag (passed through
    # llama.cpp's --jinja chat template). Cuts LLM TTFB ~6x in measurement.
    t0 = time.perf_counter()
    client = OpenAI(base_url=LLAMA_URL, api_key="local")
    resp = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    text = resp.choices[0].message.content or ""
    llm_ms = (time.perf_counter() - t0) * 1000
    tokens = resp.usage.completion_tokens if resp.usage else 0
    print(f"<<< LLM response ({llm_ms:.0f}ms, {tokens} tokens, {tokens/(llm_ms/1000):.1f} tok/s):")
    print(text)
    print()

    # --- TTS leg ---
    if not text.strip():
        print("⚠ empty LLM response — no audio generated")
        return 1

    out_file = OUT_DIR / f"demo-{int(time.time())}.mp3"
    t1 = time.perf_counter()
    comm = edge_tts.Communicate(text, TTS_VOICE)
    await comm.save(str(out_file))
    tts_ms = (time.perf_counter() - t1) * 1000
    size_kb = out_file.stat().st_size / 1024
    print(f"<<< TTS audio: {out_file} ({tts_ms:.0f}ms, {size_kb:.1f}KB)")

    # --- summary ---
    total_ms = llm_ms + tts_ms
    print(f"\n=== end-to-end: {total_ms:.0f}ms (LLM {llm_ms:.0f}ms + TTS {tts_ms:.0f}ms) ===")
    print(f"play it: afplay {out_file}")
    return 0


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Say hello and confirm the local voice pipeline is alive."
    sys.exit(asyncio.run(main(prompt)))
