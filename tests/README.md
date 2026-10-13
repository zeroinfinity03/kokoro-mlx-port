# Kokoro MLX port: tests

Checks that our MLX port (`../mlx-port/kokoro_mlx/`) gives what the official PyTorch Kokoro gives.
The port does not depend on anything here.

- `pyproject.toml`: the test environment (uv). It holds the official `kokoro` 0.9.4 with PyTorch, plus
  MLX, so one gate can put a PyTorch layer and our layer side by side.
- `cases.py`: the utterances and narration texts both models are run on.
- `capture/capture_reference.py`: runs the official model and writes `fixtures/` (not in git): every
  intermediate array of 8 utterances, the random numbers each run drew, a second run with other random
  numbers, and the pipeline's chunks, phonemes, frame counts and word times for 7 texts. Also writes
  `listening/<name>_official.wav`.
- `layers.py`: gate 1, each building block against the real PyTorch layer.
  `tests/.venv/bin/python tests/layers.py`
- `gates.py`: gates 2 to 5 against the fixtures, run in the port's own environment.
  `mlx-port/.venv/bin/python tests/gates.py [stages|chain|sound|pipeline|all]`
  The pipeline gate writes `listening/<name>_mlx.wav` beside the official file, for listening.

Run from `Kokoro/`. Results and the problems met are in `../mlx-port/NOTES.md`.
