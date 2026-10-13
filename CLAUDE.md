# Kokoro (Kokoro-82M text to speech, MLX port)

Our own MLX version of hexgrad's Kokoro-82M, written from the official PyTorch code (`repo/kokoro/`,
kokoro 0.9.4) and measured against it. Made for Vik's Guru project, which today still runs the official
PyTorch package; switching Guru to this is Vik's call.

- Code: `mlx-port/kokoro_mlx/` (Python on Apple's `mlx`; text to phonemes by `misaki`, as in the official
  package). How to use it, the layout and the measured status: `mlx-port/README.md`. What has to match
  the official code and why: `mlx-port/NOTES.md`.
- Say something: `cd mlx-port && .venv/bin/python -m kokoro_mlx "text" --voice af_heart --out out.wav`.
- The reference is never edited: `repo/` (official code), `pytorch/Kokoro-82M/` (official weights, voices).
- No code from `mlx-audio`. It has four differences from the official code (NOTES.md section 5).

## Rules

- After any change to `kokoro_mlx/`, run the gates before reporting:
  `tests/.venv/bin/python tests/layers.py && mlx-port/.venv/bin/python tests/gates.py` (from `Kokoro/`).
  The bar is the official output: identical phoneme lengths and word times, every stage within 1e-4.
- Float32 only, and `MLX_ENABLE_TF32=0` stays (set in `kokoro_mlx/__init__.py`): on Vik's M5 the default
  reduced-precision products move rounded durations.
- Weights are read from `mlx-port/models/` directly, never through links. They are made from the
  official file by `mlx-port/export/export_weights.py` and are not in git.
- A new Kokoro release: put its code in `repo/` and weights in `pytorch/`, diff against what is there,
  port the diff, re-record `tests/fixtures`, run the gates.
