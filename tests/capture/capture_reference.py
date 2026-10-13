"""Runs the OFFICIAL PyTorch Kokoro (kokoro 0.9.4, CPU, float32) and records what the gates compare against.

    cd Kokoro && tests/.venv/bin/python tests/capture/capture_reference.py

Writes tests/fixtures/:
  <case>.npz / <case>.json   one utterance: the inputs, every intermediate array of KModel.forward_with_tokens,
                             the audio, the random numbers that run drew (so MLX can replay them), and the
                             audio of a second run with other random numbers
  e2e.json                   the pipeline as an application calls it: chunks, phonemes, frame counts, word times
"""
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
TESTS = Path(__file__).resolve().parents[1]
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS))

import numpy as np
import torch
from kokoro import KModel, KPipeline

from cases import CASES, TEXTS

OFFICIAL = ROOT / "pytorch" / "Kokoro-82M"
FIX = TESTS / "fixtures"
FIX.mkdir(exist_ok=True)

model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(OFFICIAL / "config.json"), model=str(OFFICIAL / "kokoro-v1_0.pth")).eval()
pipes = {}
def pipeline(code):
    if code not in pipes:
        pipes[code] = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M", model=model)
    return pipes[code]
def voice_file(v):
    return ",".join(str(OFFICIAL / "voices" / f"{x}.pt") for x in v.split(","))


class Tape:
    """Stands in for torch.rand / torch.randn_like and keeps what was drawn."""
    def __init__(self, seed):
        self.g, self.draws = np.random.default_rng(seed), []
    def rand(self, *shape, **kw):
        a = self.g.random(shape, dtype=np.float32); self.draws.append(a); return torch.from_numpy(a)
    def randn_like(self, t, **kw):
        a = self.g.standard_normal(tuple(t.shape), dtype=np.float32); self.draws.append(a); return torch.from_numpy(a)

def with_tape(seed, fn):
    tape = Tape(seed); r0, r1 = torch.rand, torch.randn_like
    torch.rand, torch.randn_like = tape.rand, tape.randn_like
    try:
        out = fn()
    finally:
        torch.rand, torch.randn_like = r0, r1
    return out, tape


@torch.no_grad()
def traced(input_ids, ref_s, speed):
    """KModel.forward_with_tokens (repo/kokoro/model.py 86-119), statement for statement, keeping every array."""
    m, t = model, {}
    input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], dtype=torch.long)
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
    t["bert_dur"] = bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
    t["d_en"] = d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    t["d"] = d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    t["lstm_x"], _ = x, _ = m.predictor.lstm(d)
    t["duration_logits"] = duration = m.predictor.duration_proj(x)
    t["duration"] = duration = torch.sigmoid(duration).sum(axis=-1) / speed
    t["pred_dur"] = pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
    aln = torch.zeros((input_ids.shape[1], indices.shape[0])); aln[indices, torch.arange(indices.shape[0])] = 1
    aln = aln.unsqueeze(0)
    t["en"] = en = d.transpose(-1, -2) @ aln
    t["F0_pred"], t["N_pred"] = F0_pred, N_pred = m.predictor.F0Ntrain(en, s)
    t["t_en"] = t_en = m.text_encoder(input_ids, input_lengths, text_mask)
    t["asr"] = asr = t_en @ aln
    got = {}
    hooks = [m.decoder.generator.register_forward_pre_hook(lambda mod, a: got.update(gen_x=a[0])),
             m.decoder.generator.m_source.register_forward_hook(lambda mod, a, o: got.update(har_source=o[0])),
             m.decoder.generator.conv_post.register_forward_hook(lambda mod, a, o: got.update(conv_post=o))]
    try:
        t["audio"] = m.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
    finally:
        for h in hooks:
            h.remove()
    t.update(got)
    return t


for i, (name, text, voice, speed) in enumerate(CASES):
    ps, _ = pipeline(voice[0]).g2p(text)
    ids = [model.vocab[p] for p in ps if p in model.vocab]
    pack = torch.load(OFFICIAL / "voices" / f"{voice}.pt", map_location="cpu", weights_only=True)
    ref_s = pack[len(ps) - 1]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    t, tape = with_tape(1000 + i, lambda: traced(input_ids, ref_s, speed))
    (audio2, dur2), _ = with_tape(1000 + i, lambda: model.forward_with_tokens(input_ids, ref_s, speed))
    assert torch.equal(audio2, t["audio"]) and torch.equal(dur2, t["pred_dur"]), name   # the traced copy is the real thing
    out = {k: v.numpy() for k, v in t.items()}
    # the same utterance once more with other random numbers: how far the official model is from itself
    torch.manual_seed(2000 + i)
    with torch.no_grad():
        out["audio_other_run"] = model.forward_with_tokens(input_ids, ref_s, speed)[0].numpy()
    out.update({f"draw_{j}": a for j, a in enumerate(tape.draws)})
    out["input_ids"], out["ref_s"] = input_ids.numpy(), ref_s.numpy()
    np.savez(FIX / f"{name}.npz", **out)
    json.dump({"name": name, "text": text, "voice": voice, "speed": speed, "phonemes": ps}, open(FIX / f"{name}.json", "w"), ensure_ascii=False)
    print(f"{name:18s} phonemes {len(ps):3d}  frames {int(t['pred_dur'].sum()):5d}  audio {t['audio'].shape[0] / 24000:6.2f} s")

def save_wav(path, a):
    import wave
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000)
        f.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())

LISTEN = TESTS / "listening"
LISTEN.mkdir(exist_ok=True)
e2e = {}
for name, voice, speed, text in TEXTS:
    chunks, pos, audio = [], 0.0, []
    for r in pipeline(voice[0])(text, voice=voice_file(voice), speed=speed, split_pattern=None):
        n = r.audio.shape[0]
        words = [[t.text, round(pos + t.start_ts, 6), round(pos + t.end_ts, 6)] for t in r.tokens if t.start_ts is not None and t.end_ts is not None]
        chunks.append({"graphemes": r.graphemes, "phonemes": r.phonemes, "pred_dur": r.pred_dur.tolist(), "samples": n, "words": words})
        pos += n / 24000
        audio.append(r.audio.numpy())
    e2e[name] = {"voice": voice, "speed": speed, "text": text, "chunks": chunks}
    save_wav(LISTEN / f"{name}_official.wav", np.concatenate(audio))
    print(f"{name:18s} chunks {len(chunks)}  words {sum(len(c['words']) for c in chunks):3d}  audio {pos:6.2f} s")
json.dump(e2e, open(FIX / "e2e.json", "w"), ensure_ascii=False)
print("fixtures written to", FIX)
