# Port spec: read line by line, then measured

Reference: hexgrad's `kokoro` 0.9.4 (Apache 2.0), the package Vik's Guru project runs, copied to
`../repo/kokoro/`; its ALBERT is Hugging Face transformers 5.19.0 (`../repo/transformers_modeling_albert.py`);
weights `hexgrad/Kokoro-82M` `kokoro-v1_0.pth` (sha256 `496dba11…ad1e4`) in `../pytorch/Kokoro-82M/`.
The official model runs in float32 on the CPU (Guru's setting), and so does everything measured here.

## 1. What Kokoro is

Phonemes in, 24 kHz audio out, 81.8 M parameters in five parts (the five state dicts of the .pth):

| Part | Tensors | What it does |
|---|---|---|
| `bert` | 25 | phoneme-level ALBERT: 12 passes through one shared layer, hidden 768 |
| `bert_encoder` | 2 | Linear 768 → 512 |
| `predictor` | 122 | duration encoder (3 x LSTM + AdaLayerNorm), duration head, pitch (F0) and energy (N) curves |
| `text_encoder` | 24 | embedding, 3 x (conv, LayerNorm, LeakyReLU), LSTM |
| `decoder` | 375 | AdaIN residual blocks, then the iSTFTNet generator: harmonic source from F0, two upsamplers (x10, x6), inverse STFT (n_fft 20, hop 5) |

One frame is 600 samples (25 ms). A voice is a table [510, 1, 256]: row `len(phonemes) - 1` is the
style; its first 128 numbers go to the decoder, the last 128 to the predictor.

## 2. Where each official line went

| Official (`../repo/kokoro/`) | Ours (`kokoro_mlx/`) |
|---|---|
| `model.py` 31-75 `KModel.__init__`, loading | `model.py` `KModel.__init__` |
| `model.py` 86-119 `forward_with_tokens` | `model.py` `forward_with_tokens` |
| `model.py` 121-137 `forward` | `model.py` `__call__` |
| `modules.py` 11-18 LinearNorm, 21-32 LayerNorm | `modules.py` `duration_proj`, inline in `TextEncoder` |
| `modules.py` 35-69 TextEncoder | `modules.py` `TextEncoder` |
| `modules.py` 72-88 AdaLayerNorm | `modules.py` `ada_layer_norm` |
| `modules.py` 91-134 ProsodyPredictor (inference path) | `modules.py` `ProsodyPredictor` |
| `modules.py` 137-176 DurationEncoder | `modules.py` `DurationEncoder` |
| `modules.py` 180-183 CustomAlbert + transformers AlbertModel | `albert.py` |
| `istftnet.py` 20-31 AdaIN1d | `istftnet.py` `adain` |
| `istftnet.py` 34-77 AdaINResBlock1 | `istftnet.py` `AdaINResBlock1` |
| `istftnet.py` 80-105 TorchSTFT | `istftnet.py` `STFT` |
| `istftnet.py` 108-209 SineGen, 212-254 SourceModuleHnNSF | `istftnet.py` `SineGen`, `Generator.source` |
| `istftnet.py` 257-325 Generator | `istftnet.py` `Generator` |
| `istftnet.py` 328-381 UpSample1d, AdainResBlk1d | `nn.py` `upsample_nearest`, `istftnet.py` `AdainResBlk1d` |
| `istftnet.py` 384-421 Decoder | `istftnet.py` `Decoder` |
| `pipeline.py` (all) KPipeline | `pipeline.py` |
| torch layers: Conv1d, ConvTranspose1d, weight_norm, LSTM, layer_norm, InstanceNorm1d, F.interpolate | `nn.py` |

Not ported, because inference never reaches them: `ProsodyPredictor.forward` (the training path),
`SineGen` with `flag_for_pulse`, `CustomSTFT` (`disable_complex=True`, the ONNX path), `KModelForONNX`,
the ALBERT pooler.

## 3. What has to be exactly right (each one is a gate)

1. **Weights stay as the official file has them.** The export only renames `module.x` to `<part>.x`.
   Layout changes happen at load, in `nn.py`: Conv1d (out, in, k) → (out, k, in); ConvTranspose1d
   (in, out, k) → (out, k, in), and the depthwise one (C, 1, k) → (C, k, 1).
2. **weight_norm**: `w = v * (g / ||v||)`, the norm over every axis but the first, no epsilon. For a
   ConvTranspose1d the first axis is the input channel. 89 layers use it; it is folded once at load.
3. **InstanceNorm in AdaIN has no learned scale.** The official layer is built `affine=True`, but the
   checkpoint has no such tensors and the official loader is not strict (every key carries `module.`,
   so `model.py` 70-75 always takes the `strict=False` branch): they stay 1 and 0.
4. **ALBERT's activation is `gelu_new`**, the tanh form: the plbert config names no activation, so
   Hugging Face's default applies. The exact erf GELU differs by up to 5e-4 per call.
5. **LSTM**: gate order i, f, g, o; `W_ih x + b_ih + W_hh h + b_hh`; the backward direction's outputs stay
   at their own time steps; output is [forward, backward].
6. **Durations**: `round(sigmoid(logits).sum(-1) / speed)` with round-half-to-even, at least 1. There is
   no upper limit in the official code and none here.
7. **Alignment**: the official 0/1 matrix product copies feature column `i` `pred_dur[i]` times. Done
   here as a gather, which gives the same numbers without a product.
8. **Harmonic source resampling** (`F.interpolate`, linear, `align_corners=False`): output length is
   `floor(L * scale)`, rounded DOWN; the read position is `ratio * (dst + 0.5) - 0.5` with
   `ratio = float32(1 / scale)`, one rounding (PyTorch's arm64 build fuses the multiply and subtract).
9. **Nearest upsampling** by 300 and by 2 is an exact repeat.
10. **STFT / inverse STFT**: periodic Hann, centred with reflect padding (edge not repeated), no
    normalization; the inverse divides by the overlap-added squared window and drops n_fft/2 at both ends.
    The phase of a bin with zero imaginary part and negative real part is +pi.
11. **ReflectionPad1d((1, 0))** before the last skip add: the new first sample is a copy of sample 1.
12. **Slopes**: LeakyReLU 0.2 in the blocks, 0.1 inside the generator loop, 0.01 (PyTorch's default)
    before `conv_post`. Residual blocks end with `* 1/sqrt(2)`.
13. **Random numbers**: start phases uniform [0, 1) for harmonics 2 to 9 (the fundamental gets 0), added
    at the first sample only; source noise standard normal. The official code draws a third array for a
    noise branch whose output is never used; it is not drawn here.
14. **Pipeline**: misaki `en.G2P(trf=False, british=…, fallback=EspeakFallback, unk='')`; chunks of at most
    510 phonemes cut at the last `!.?…`, then `:;`, then `,—`; phonemes missing from the vocabulary are
    dropped; ids get a 0 at both ends; style row `len(ps) - 1`; word times in half-frames over 80.

## 4. Precision

Everything is float32, like the official model. On Apple's newest GPUs MLX runs float32 matrix products
on the Neural Accelerator in reduced (TF32-class) precision unless `MLX_ENABLE_TF32=0` is set; the error
is about 8e-4 relative (CPU: 4e-7). For Kokoro that is not harmless: durations are rounded to whole
frames, and in the community package it moved 1 to 3 phoneme lengths per 1,000 across the boundary.
`kokoro_mlx/__init__.py` sets `MLX_ENABLE_TF32=0` before MLX is used. Measured cost for Kokoro: none.
Measured with our own code and the default forced back on (`MLX_ENABLE_TF32=1`): 4 of 2,184 phoneme
lengths differ in the pipeline gate and word times are off by up to 50 ms; with it off, 0 and 0.
If a program imports and uses MLX before it imports `kokoro_mlx`, it has to set the variable itself.

## 5. Why our own port (the community one, checked 2026-10-12)

`mlx-community/Kokoro-82M-bf16` with `mlx-audio` 0.5.8 (Prince Canuma's package; not Apple's):
- Weights: identical to the official ones, and float32 despite the repo's name.
- Code, four differences: exact GELU in ALBERT (item 4); `interpolate` rounds the length up, so for 27%
  of clip lengths the harmonic source lags one frame in the second half (item 8); the reflection pad is
  a zero pad (item 11); the phase of the last STFT bin comes out -pi on the M5 GPU (item 10). Plus the
  TF32 default (section 4).
- Result as shipped: 3 of 2,107 phoneme lengths differ from the official ones, word times off by up to
  25 ms, spectrum 0.99 dB from the official audio where the official is 0.48 dB from itself.
- Its API also differs from the official one (class names, voices as `.safetensors` looked up by repo
  id, audio shape (1, N), no word times through `generate`).

## 6. Results

See `README.md` "Status". In one line: every stage within 4e-6 of the official output, 0 of 3,257
phoneme lengths different (1,073 in the chain gate, 2,184 in the pipeline gate), every word time
identical, 17x realtime.

## 7. Problems met

1. **Resampling was one float32 step off for some samples** (relative 7e-5 at 28 s, growing with length).
   Rounding the product and the subtraction separately, as MLX's GPU does, differs from PyTorch's fused
   multiply-subtract. Fix: the read positions are worked out on the CPU as an exact double product rounded
   once to float32 (`nn.interpolate_linear`). After it: 1.0e-7 at every length.
2. **The waveform cannot match better than about 24 dB on a 28 s clip.** The phase is a running sum in
   float32; a different summation order gives a slightly different phase late in the clip. The gate for
   audio is therefore SNR > 20 dB next to exact lengths and F0, not sample equality.
3. **`misaki` imports PyTorch**, so PyTorch has to be installed for the text side even though the model
   code never loads it (checked: after loading the model and making audio, `torch` is not in
   `sys.modules`).
