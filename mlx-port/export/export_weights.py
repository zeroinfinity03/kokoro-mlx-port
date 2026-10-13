"""Official Kokoro weights and voices -> safetensors files our MLX code reads.

Run once, with the tests environment (it has PyTorch):
    cd Kokoro && tests/.venv/bin/python mlx-port/export/export_weights.py

Reads   pytorch/Kokoro-82M/kokoro-v1_0.pth, config.json, voices/*.pt   (hexgrad/Kokoro-82M)
Writes  mlx-port/models/kokoro-v1_0.safetensors   every tensor, same name, same layout, same numbers
        mlx-port/models/voices.safetensors        one tensor per voice, [510, 1, 256]
        mlx-port/models/config.json

Nothing is converted here: the tensors keep PyTorch's names and layouts, so the file can be compared
with the official one bit for bit. Layout changes (weight norm, conv axes) happen when the model loads
(kokoro_mlx/nn.py), where they sit next to the code that uses them.
"""
import hashlib
import shutil
from pathlib import Path

import torch
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pytorch" / "Kokoro-82M"
DST = ROOT / "mlx-port" / "models"
OFFICIAL_SHA256 = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"  # kokoro-v1_0.pth on Hugging Face

pth = SRC / "kokoro-v1_0.pth"
assert hashlib.sha256(pth.read_bytes()).hexdigest() == OFFICIAL_SHA256, "not the official kokoro-v1_0.pth"
DST.mkdir(parents=True, exist_ok=True)

# The official file holds five state dicts (bert, bert_encoder, predictor, decoder, text_encoder), every
# key with a "module." prefix that the official loader strips (repo/kokoro/model.py 68-75).
weights = {}
for part, sd in torch.load(pth, map_location="cpu", weights_only=True).items():
    for k, v in sd.items():
        assert k.startswith("module."), k
        weights[f"{part}.{k[7:]}"] = v.numpy()
save_file(weights, str(DST / "kokoro-v1_0.safetensors"))

voices = {f.stem: torch.load(f, map_location="cpu", weights_only=True).numpy() for f in sorted((SRC / "voices").glob("*.pt"))}
save_file(voices, str(DST / "voices.safetensors"))
shutil.copyfile(SRC / "config.json", DST / "config.json")

# read back and compare with what PyTorch holds
back = load_file(str(DST / "kokoro-v1_0.safetensors"))
assert set(back) == set(weights) and all(back[k].dtype == weights[k].dtype and back[k].tobytes() == weights[k].tobytes() for k in weights)
vback = load_file(str(DST / "voices.safetensors"))
assert set(vback) == set(voices) and all(vback[k].tobytes() == voices[k].tobytes() for k in voices)
n = sum(v.size for v in weights.values())
print(f"weights: {len(weights)} tensors, {n:,} numbers, types {sorted({str(v.dtype) for v in weights.values()})}: identical after reading back")
print(f"voices : {len(voices)}, each {next(iter(voices.values())).shape}: identical after reading back")
