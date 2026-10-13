# kokoro-mlx-port

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text to speech on Apple silicon with
[MLX](https://github.com/ml-explore/mlx), ported line by line from hexgrad's official PyTorch package
(`kokoro` 0.9.4) and measured against it.

- **Same output as the official package:** every phoneme length and every word time identical
  (0 of 3,257 phoneme lengths and 0 of 450 word times different in the gates), every model stage
  within 4e-6 of the official one, and a spectrum as close to the official audio as a second official run.
- **Fast:** 15–18x realtime on an Apple M5 GPU.
- **Same API:** `KPipeline` with the official arguments; `result.audio` is a numpy array.

```bash
cd mlx-port && uv sync
.venv/bin/python -m kokoro_mlx "Hello world." --voice af_heart --out hello.wav --words hello.json
```

Weights are not in this repository: `mlx-port/export/export_weights.py` makes them from the official
`kokoro-v1_0.pth`. Full setup, layout and measured status: [`mlx-port/README.md`](mlx-port/README.md).
What has to match the official code and why: [`mlx-port/NOTES.md`](mlx-port/NOTES.md). The tests and
gates: [`tests/`](tests/).

Apache License 2.0; see `LICENSE` and `NOTICE`.
