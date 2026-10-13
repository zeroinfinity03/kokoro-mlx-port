"""Say something:

    cd Kokoro/mlx-port && .venv/bin/python -m kokoro_mlx "Hello world." --voice af_heart --out hello.wav

Writes a 24 kHz 16-bit WAV and, with --words, a JSON list of every word with its start and end time.
The language comes from the voice name: a* is American English, b* British.
"""
import argparse
import json
import time
import wave

import numpy as np

from . import SAMPLE_RATE, KPipeline


def main():
    p = argparse.ArgumentParser(prog="kokoro_mlx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text")
    p.add_argument("--voice", default="af_heart", help="one voice, or several separated by commas to blend them")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", default="out.wav")
    p.add_argument("--words", help="also write the word times to this JSON file")
    a = p.parse_args()

    pipeline = KPipeline(lang_code=a.voice[0])
    t0 = time.time()
    audio, words, position = [], [], 0.0
    for result in pipeline(a.text, voice=a.voice, speed=a.speed, split_pattern=None):
        for t in result.tokens or []:
            if t.start_ts is not None and t.end_ts is not None:
                words.append({"word": t.text, "start": round(position + t.start_ts, 3), "end": round(position + t.end_ts, 3)})
        audio.append(result.audio)
        position += len(result.audio) / SAMPLE_RATE
    audio = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
    with wave.open(a.out, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SAMPLE_RATE)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    if a.words:
        with open(a.words, "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False, indent=1)
    dt = time.time() - t0
    print(f"{a.out}: {position:.2f} s of speech, {len(words)} words, made in {dt:.2f} s ({position / max(dt, 1e-9):.0f}x realtime)")


if __name__ == "__main__":
    main()
