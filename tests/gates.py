"""Gates 2-4: our MLX Kokoro against what the official PyTorch Kokoro produced (tests/fixtures).

    cd Kokoro && mlx-port/.venv/bin/python tests/gates.py [stages|chain|sound|pipeline|all]

stages    every stage fed the OFFICIAL input for that stage, so a mistake in one stage cannot hide behind,
          or be blamed on, another
chain     the whole model from phoneme ids, with the official run's random numbers replayed: every phoneme
          length must be identical, and the waveform must match
sound     the model run freely, with its own random numbers, as in real use: its spectrum must be as close
          to the official audio as a second official run is
pipeline  text in, as an application calls it: chunks, phonemes, frame counts and word times must be identical;
          also writes tests/listening/<name>_mlx.wav next to the official <name>_official.wav
"""
import json
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / "mlx-port"))

import kokoro_mlx  # noqa: E402  (sets full-precision MLX before MLX is used)
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from kokoro_mlx.nn import linear  # noqa: E402

FIX = TESTS / "fixtures"
REL = 1e-4          # stage outputs: relative to the largest value in the official array
SNR_SOURCE = 25.0   # dB, harmonic source; float32 phase accumulation is the limit, worst on the longest clip
SNR_AUDIO = 20.0    # dB, final waveform
fails = 0


def rel(ref, got):
    ref = np.squeeze(np.asarray(ref)).astype(np.float64)
    got = np.squeeze(np.asarray(got)).astype(np.float64)
    assert ref.shape == got.shape, (ref.shape, got.shape)
    return float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-12))


def snr(ref, got):
    ref = np.squeeze(np.asarray(ref)).astype(np.float64)
    got = np.squeeze(np.asarray(got)).astype(np.float64)
    assert ref.shape == got.shape, (ref.shape, got.shape)
    return float(10 * np.log10((ref ** 2).sum() / max(((ref - got) ** 2).sum(), 1e-30)))


def verdict(ok):
    global fails
    fails += not ok
    return "PASS" if ok else "FAIL"


def cases():
    for f in sorted(FIX.glob("*.npz")):
        yield f.stem, np.load(f), json.loads(f.with_suffix(".json").read_text())


def gate_stages(model):
    print("GATE 2 stages: each stage fed the official input, worst relative difference per stage")
    names = ["bert", "bert_encoder", "duration encoder", "lstm", "duration_proj", "F0", "N", "text_encoder",
             "decoder features", "harmonic source (SNR dB)", "generator audio (SNR dB)"]
    worst = {n: 0.0 for n in names}
    worst["harmonic source (SNR dB)"] = worst["generator audio (SNR dB)"] = 1e9
    for name, o, meta in cases():
        A = lambda k: mx.array(o[k])
        ids, ref_s = A("input_ids"), A("ref_s")
        s, sd = ref_s[:, 128:], ref_s[:, :128]
        p, r = model.predictor, {}
        r["bert"] = rel(o["bert_dur"], model.bert(ids))
        r["bert_encoder"] = rel(o["d_en"], linear(A("bert_dur"), model.weights, "bert_encoder").transpose(0, 2, 1))
        r["duration encoder"] = rel(o["d"], p.text_encoder(A("d_en"), s))
        r["lstm"] = rel(o["lstm_x"], p.lstm(A("d")))
        r["duration_proj"] = rel(o["duration_logits"], p.duration_proj(A("lstm_x")))
        F0, N = p.F0Ntrain(A("en"), s)
        r["F0"], r["N"] = rel(o["F0_pred"], F0), rel(o["N_pred"], N)
        r["text_encoder"] = rel(o["t_en"], model.text_encoder(ids))
        r["decoder features"] = rel(o["gen_x"], model.decoder.features(A("asr"), A("F0_pred"), A("N_pred"), sd))
        draws = (A("draw_0"), A("draw_1"))
        gen = model.decoder.generator
        hs = snr(o["har_source"], gen.source(A("F0_pred"), draws))
        ga = snr(o["audio"], gen(A("gen_x"), sd, A("F0_pred"), draws))
        for n in names[:-2]:
            worst[n] = max(worst[n], r[n])
        worst[names[-2]], worst[names[-1]] = min(worst[names[-2]], hs), min(worst[names[-1]], ga)
        print(f"  {name:18s} " + " ".join(f"{r[n]:.1e}" for n in names[:-2]) + f"  source {hs:5.1f} dB  audio {ga:5.1f} dB")
    for n in names[:-2]:
        print(f"  {verdict(worst[n] < REL)}  {n:26s} {worst[n]:.1e}  (gate < {REL:.0e})")
    print(f"  {verdict(worst[names[-2]] > SNR_SOURCE)}  {names[-2]:26s} lowest {worst[names[-2]]:.1f}  (gate > {SNR_SOURCE:.0f})")
    print(f"  {verdict(worst[names[-1]] > SNR_AUDIO)}  {names[-1]:26s} lowest {worst[names[-1]]:.1f}  (gate > {SNR_AUDIO:.0f})")


def gate_chain(model):
    print("GATE 3 chain: phoneme ids to audio, official random numbers replayed")
    total = bad = 0
    for name, o, meta in cases():
        t = {}
        t0 = time.time()
        audio, pred_dur = model.forward_with_tokens(o["input_ids"], o["ref_s"], meta["speed"],
                                                    draws=(mx.array(o["draw_0"]), mx.array(o["draw_1"])), trace=t)
        dt = time.time() - t0
        od = np.squeeze(o["pred_dur"])
        n = int((od != pred_dur).sum())
        total += od.size
        bad += n
        ddur = float(np.abs(np.squeeze(o["duration"]) - np.squeeze(np.array(t["duration"]))).max())
        if n == 0:
            f0 = float(np.abs(np.squeeze(o["F0_pred"]) - np.squeeze(np.array(t["F0_pred"]))).max())
            a = snr(o["audio"], audio)
            ok = a > SNR_AUDIO
            print(f"  {verdict(ok)}  {name:18s} {od.size:3d} lengths identical ({int(od.sum())} frames), largest duration difference "
                  f"{ddur:.5f} frame, F0 within {f0:.4f} Hz, audio SNR {a:.1f} dB, {audio.size / 24000:.1f} s in {dt:.2f} s")
        else:
            print(f"  {verdict(False)}  {name:18s} {n} of {od.size} phoneme lengths differ ({int(od.sum())} vs {int(pred_dur.sum())} frames)")
    print(f"  {verdict(bad == 0)}  phoneme lengths that differ from the official ones: {bad} of {total}")


def log_mel(a, n_fft=1024, hop=256, n_mels=80, sr=24000):
    a = np.squeeze(a).astype(np.float64)
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.stack([np.clip(np.minimum((bins - pts[i]) / (pts[i + 1] - pts[i]), (pts[i + 2] - bins) / (pts[i + 2] - pts[i + 1])), 0, None) for i in range(n_mels)])
    win = np.hanning(n_fft + 1)[:-1]
    frames = np.stack([a[i * hop:i * hop + n_fft] * win for i in range(1 + (len(a) - n_fft) // hop)])
    return 10 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) ** 2 @ fb.T + 1e-10)


def spectrum_distance(ref, got):
    """Mean difference of the two log-mel spectra in dB, over the frames that are not near-silent."""
    A, B = log_mel(ref), log_mel(got)
    loud = A.max(axis=1) > A.max() - 50
    return float(np.abs(A[loud] - B[loud]).mean())


def gate_sound(model):
    print("GATE 5 sound: run freely with its own random numbers; spectrum difference from the official audio (dB)")
    mx.random.seed(7)
    ours, theirs = [], []
    for name, o, meta in cases():
        audio, pred_dur = model.forward_with_tokens(o["input_ids"], o["ref_s"], meta["speed"])
        assert np.array_equal(pred_dur, np.squeeze(o["pred_dur"]))
        a, b = spectrum_distance(o["audio"], audio), spectrum_distance(o["audio"], o["audio_other_run"])
        ours.append(a); theirs.append(b)
        print(f"  {name:18s} ours vs official {a:.2f}    official vs official (another run) {b:.2f}")
    ok = np.mean(ours) <= np.mean(theirs) * 1.15
    print(f"  {verdict(ok)}  average: ours {np.mean(ours):.2f} dB, official against itself {np.mean(theirs):.2f} dB  (gate: no more than 15% above)")


def save_wav(path, a):
    import wave
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000)
        f.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())


def gate_pipeline(model):
    from kokoro_mlx import KPipeline
    print("GATE 4 pipeline: text in, the way an application calls it")
    ref = json.loads((FIX / "e2e.json").read_text())
    pipes, words, nlen, bad_len, worst = {}, 0, 0, 0, 0.0
    for name, r in ref.items():
        code = r["voice"][0]
        pipe = pipes.setdefault(code, KPipeline(lang_code=code, model=model))
        chunks, pos, t0, audio = [], 0.0, time.time(), []
        for res in pipe(r["text"], voice=r["voice"], speed=r["speed"], split_pattern=None):
            n = res.audio.shape[0]
            audio.append(res.audio)
            w = [[t.text, round(pos + t.start_ts, 6), round(pos + t.end_ts, 6)] for t in res.tokens if t.start_ts is not None and t.end_ts is not None]
            chunks.append({"graphemes": res.graphemes, "phonemes": res.phonemes, "pred_dur": [int(x) for x in res.pred_dur], "samples": n, "words": w})
            pos += n / 24000
        dt = time.time() - t0
        (TESTS / "listening").mkdir(exist_ok=True)
        save_wav(TESTS / "listening" / f"{name}_mlx.wav", np.concatenate(audio))
        same_text = [c["graphemes"] for c in chunks] == [c["graphemes"] for c in r["chunks"]]
        same_ph = [c["phonemes"] for c in chunks] == [c["phonemes"] for c in r["chunks"]]
        od = [d for c in r["chunks"] for d in c["pred_dur"]]
        md = [d for c in chunks for d in c["pred_dur"]]
        ndiff = sum(a != b for a, b in zip(od, md)) if len(od) == len(md) else len(od)
        ow = [w for c in r["chunks"] for w in c["words"]]
        mw = [w for c in chunks for w in c["words"]]
        same_words = [w[0] for w in ow] == [w[0] for w in mw]
        tdiff = max([abs(a[1] - b[1]) for a, b in zip(ow, mw)] + [abs(a[2] - b[2]) for a, b in zip(ow, mw)] + [0.0]) * 1000 if same_words else float("inf")
        same_n = [c["samples"] for c in chunks] == [c["samples"] for c in r["chunks"]]
        ok = same_text and same_ph and ndiff == 0 and same_words and tdiff == 0 and same_n
        words += len(ow); nlen += len(od); bad_len += ndiff; worst = max(worst, tdiff)
        print(f"  {verdict(ok)}  {name:14s} chunks {len(chunks)}/{len(r['chunks'])}  phonemes {'same' if same_ph else 'DIFFERENT'}  "
              f"lengths differing {ndiff}/{len(od)}  words {len(mw)}/{len(ow)}  largest word-time difference {tdiff:.1f} ms  "
              f"{pos:.1f} s in {dt:.2f} s ({pos / dt:.0f}x realtime)")
    print(f"  {verdict(bad_len == 0 and worst == 0)}  {words} words, {nlen} phoneme lengths: {bad_len} differ, largest word-time difference {worst:.1f} ms")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = kokoro_mlx.KModel()
    model_needs_torch = "torch" in sys.modules          # checked before misaki (text to phonemes) is imported; misaki loads it
    if which in ("stages", "all"):
        gate_stages(model)
    if which in ("chain", "all"):
        gate_chain(model)
    if which in ("sound", "all"):
        gate_sound(model)
    if which in ("pipeline", "all"):
        gate_pipeline(model)
    print("peak memory: %.2f GB;  PyTorch loaded by the model code: %s;  mlx-audio loaded: %s" % (
        mx.get_peak_memory() / 1e9, model_needs_torch, "mlx_audio" in sys.modules))
    print("GATES:", "ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
