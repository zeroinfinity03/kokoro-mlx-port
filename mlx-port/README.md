# Kokoro-82M → MLX (Python)

Goal: run hexgrad's Kokoro-82M text to speech on Apple MLX with our own code, and get what the
official PyTorch package (`kokoro` 0.9.4) gets: the same phoneme lengths, the same word times, the
same sound. Same method as the ACE-Step, YuE and Stable Audio ports: read every line of the reference
first (`NOTES.md`), keep the official weights as they are, record fixtures from the real PyTorch
model, then port one piece at a time, each behind a measured gate.

Python, not Swift, because Kokoro's text side is the Python package `misaki` (text to phonemes), which
the official code uses too. The model code itself needs only `mlx` and `numpy`.

No code is taken from `mlx-audio` (the community package that also has a Kokoro). We checked that one
first (2026-10-12): correct weights, four places where its code differs from the official one, see
`NOTES.md` "Why our own port".

## Use

```bash
cd Kokoro/mlx-port
.venv/bin/python -m kokoro_mlx "Hello world." --voice af_heart --out hello.wav --words hello.json
```

```python
from kokoro_mlx import KPipeline

pipeline = KPipeline(lang_code="a")            # 'a' American English, 'b' British (the voice's first letter)
for result in pipeline(text, voice="af_heart", speed=1.0, split_pattern=None):
    result.audio       # numpy float32, 24,000 Hz
    result.tokens      # every word: .text, .start_ts, .end_ts (seconds inside this chunk)
    result.pred_dur    # frames of 600 samples for <bos>, each phoneme, <eos>
```

`KPipeline` has the official class's methods and arguments. Three differences: the model and voices are
read from `models/` (nothing is downloaded), `result.audio` and `result.pred_dur` are numpy arrays
(the official ones are torch tensors, so `result.audio.numpy()` becomes `result.audio`), and
`repo_id` / `device` are accepted and ignored.

## Layout

```
mlx-port/
├── kokoro_mlx/        the port, 1,073 lines
│   ├── nn.py          the PyTorch layers Kokoro uses, for MLX (conv, transposed conv, LSTM, norms, resampling)
│   ├── albert.py      the phoneme-level ALBERT
│   ├── modules.py     text encoder, duration encoder, prosody predictor
│   ├── istftnet.py    decoder: AdaIN blocks, harmonic source, generator, STFT and inverse STFT
│   ├── model.py       KModel: phoneme ids + voice style -> audio + phoneme lengths
│   ├── pipeline.py    KPipeline: text -> phonemes (misaki) -> chunks -> audio + word times
│   └── __main__.py    the command-line tool
├── export/export_weights.py   official .pth and voices -> models/ (run once, needs PyTorch)
├── models/            kokoro-v1_0.safetensors, voices.safetensors, config.json (not in git)
├── pyproject.toml     mlx 0.32.3, numpy, misaki[en] 0.9.4 (uv)
├── NOTES.md           the reference read line by line, what has to match, results, problems met
└── README.md
```

The reference lives one level up: `../repo/` (the official code), `../pytorch/Kokoro-82M/` (the
official weights and voices), and the tests in `../tests/`.

## Setting it up from nothing

```bash
cd Kokoro
# official weights: hexgrad/Kokoro-82M on Hugging Face -> pytorch/Kokoro-82M/ (kokoro-v1_0.pth, config.json, voices/*.pt)
(cd tests && uv sync)                                    # the official PyTorch Kokoro, for the export and the reference
(cd mlx-port && uv sync)                                 # our runtime
tests/.venv/bin/python mlx-port/export/export_weights.py
tests/.venv/bin/python tests/capture/capture_reference.py
tests/.venv/bin/python tests/layers.py && mlx-port/.venv/bin/python tests/gates.py
```

## Milestones

| # | Piece | Gate |
|---|---|---|
| 0 | weights and voices exported | every tensor present, same name, same numbers as the official file |
| 1 | layers (`nn.py`) | each against the real PyTorch layer, random weights: relative difference < 2e-5 |
| 2 | every stage, fed the official input | relative difference < 1e-4; source SNR > 25 dB; audio SNR > 20 dB |
| 3 | whole model, official random numbers replayed | every phoneme length identical; audio SNR > 20 dB |
| 4 | pipeline, text in | chunks, phonemes, lengths and word times identical |
| 5 | sound, own random numbers | spectrum as close to the official audio as a second official run |

## Status (2026-10-12, Apple M5, MLX 0.32.3): all gates pass

- **M0 weights.** 548 tensors, 81,763,410 numbers, float32: identical to `kokoro-v1_0.pth` bit for bit
  (the export keeps names and layouts; the file's sha256 is checked against the official one). 54 voices,
  each [510, 1, 256], identical.
- **M1 layers: PASS**, 26 checks, worst 4.8e-7. Includes four lengths of resampling that a round-up
  would get wrong, and the depthwise transposed conv with `output_padding`.
- **M2 stages: PASS**, 8 utterances from 1.6 s to 28 s, 5 voices, speeds 0.8 to 1.25. Worst relative
  difference per stage: ALBERT 2.3e-6, bert_encoder 1.3e-6, duration encoder 1.3e-6, LSTM 1.4e-6,
  duration_proj 1.2e-6, F0 9.2e-7, N 9.0e-7, text encoder 2.1e-6, decoder features 3.7e-6. Harmonic
  source SNR 30.4 to 64.3 dB, generator audio 24.6 to 38.3 dB (lowest on the 28 s clip, where float32
  phase accumulation is the limit: the official float32 result is itself about 34 dB from a float64 one).
- **M3 chain: PASS.** 1,073 phoneme lengths, 0 differ. Predicted durations within 0.00003 frame before
  rounding, F0 within 0.006 Hz, audio SNR 23.8 to 40.1 dB.
- **M4 pipeline: PASS.** 7 texts (one split into 3 chunks, one British, one blend of two voices, speeds
  0.9 to 1.15): same chunks, same phonemes, 2,184 phoneme lengths with 0 different, 450 words with every
  start and end time identical to the official ones.
- **M5 sound: PASS.** Mean log-mel difference from the official audio 0.48 dB; the official model
  against a second run of itself 0.48 dB.
- **Speed and memory.** 59.7 s of speech in 3.46 s, 17x realtime, peak memory 3.3 GB. The official
  package on this Mac's CPU: 12x realtime.
- The model code loads neither PyTorch nor mlx-audio. `misaki` (text to phonemes) imports PyTorch, as it
  does under the official package, so PyTorch still has to be installed.
