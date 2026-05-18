#!/usr/bin/env python3
"""Verify the install — imports + model loading, no audio I/O yet.

Catches missing wheels, wrong Python version, and weight files that didn't download.
Run after install.sh. Exit 0 = pass, non-zero = pass not ready.
"""
from __future__ import annotations

import sys
from pathlib import Path


def check(label: str, fn) -> bool:
    print(f"  {label:35s}", end=" ", flush=True)
    try:
        result = fn()
        print(f"✓ {result if result else ''}")
        return True
    except Exception as e:
        print(f"✗ {type(e).__name__}: {str(e)[:80]}")
        return False


def main() -> int:
    print("\n=== imports ===")
    checks = [
        ("python version", lambda: sys.version.split()[0]),
        ("pipecat", lambda: __import__("pipecat").__version__ if hasattr(__import__("pipecat"), "__version__") else "imported"),
        ("pipecat.frames", lambda: __import__("pipecat.frames", fromlist=["Frame"])),
        ("pipecat.pipeline", lambda: __import__("pipecat.pipeline", fromlist=["pipeline"])),
        ("faster_whisper", lambda: __import__("faster_whisper").__version__),
        ("edge_tts", lambda: __import__("edge_tts").__version__),
        ("openai (client)", lambda: __import__("openai").__version__),
        ("soundfile", lambda: __import__("soundfile").__version__),
        ("neutts.NeuTTS", lambda: __import__("neutts", fromlist=["NeuTTS"]).NeuTTS and "imported"),
    ]

    ok = True
    for label, fn in checks:
        ok &= check(label, fn)

    print("\n=== model files ===")
    root = Path(__file__).resolve().parent.parent
    models = [
        ("Qwen3-8B-UD-Q5_K_XL.gguf", "~5.5 GB (symlink to turboquant fork)"),
    ]
    for name, expected in models:
        path = root / "models" / name
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"  {name:45s} ✓ {size_gb:.2f} GB on disk")
        else:
            print(f"  {name:45s} ✗ missing (expected {expected})")
            ok = False

    print("\n=== runtime probes ===")
    ok &= check(
        "faster_whisper.WhisperModel('base')",
        lambda: __import__("faster_whisper", fromlist=["WhisperModel"]).WhisperModel(
            "base", device="cpu", compute_type="int8"
        ) and "loaded",
    )
    # NeuTTS instantiation downloads ~750MB of weights on first call — skip in
    # smoke-test (verify import only above). Run pipeline.py to actually load.

    print()
    if ok:
        print("✅ smoke test PASSED — environment is ready for pipeline.py")
        return 0
    else:
        print("❌ smoke test FAILED — fix the ✗ items above and re-run install.sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
