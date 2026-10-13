"""Kokoro-82M text to speech on Apple MLX: our own port of hexgrad's official PyTorch code (kokoro 0.9.4).

    from kokoro_mlx import KPipeline
    pipeline = KPipeline(lang_code="a")
    for result in pipeline("Hello world.", voice="af_heart", speed=1.0, split_pattern=None):
        result.audio      # numpy float32, 24,000 Hz
        result.tokens     # each word: .text, .start_ts, .end_ts
"""
import os

# On Apple's newest GPUs MLX runs 32-bit matrix products at reduced (TF32) precision unless told
# otherwise. Kokoro rounds its predicted durations to whole frames, and that small error is enough to
# move one across a rounding boundary, so the official lengths and word times would not be reproduced.
# It has to be set before MLX does its first product; it costs Kokoro no speed.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

from .model import SAMPLE_RATE, KModel  # noqa: E402

__all__ = ["KModel", "KPipeline", "SAMPLE_RATE"]


def __getattr__(name):
    # the pipeline needs misaki (text to phonemes); the model alone does not
    if name == "KPipeline":
        from .pipeline import KPipeline
        return KPipeline
    raise AttributeError(name)
