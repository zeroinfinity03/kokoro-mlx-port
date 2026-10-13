"""Gate 1: every building block in kokoro_mlx/nn.py against the PyTorch layer it stands for.

Random weights and inputs, the real PyTorch layer on the CPU, our layer on MLX, numbers compared.
Run with the tests environment (it has both):   tests/.venv/bin/python tests/layers.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mlx-port"))

import mlx.core as mx
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from kokoro_mlx import nn

torch.manual_seed(0)
TOL = 2e-5
fails = 0


def sd(module, name):
    return {f"{name}.{k}": mx.array(v.numpy()) for k, v in module.state_dict().items()}


def check(label, ref, got, tol=TOL):
    global fails
    ref = ref.detach().numpy().astype(np.float64)
    got = np.array(got).astype(np.float64)
    if ref.shape != got.shape:
        print(f"  FAIL  {label}: shape {got.shape}, want {ref.shape}")
        fails += 1
        return
    rel = np.abs(ref - got).max() / (np.abs(ref).max() + 1e-12)
    ok = rel <= tol
    fails += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: shape {ref.shape}, max rel diff {rel:.1e}")


def randomize(m):
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    return m


with torch.no_grad():
    x = torch.randn(2, 16, 50)
    m = randomize(weight_norm(tnn.Conv1d(16, 24, 5, padding=2)))
    check("Conv1d, weight norm, k5 p2", m(x), nn.Conv1d(sd(m, "c"), "c", padding=2)(mx.array(x.numpy())))
    m = randomize(weight_norm(tnn.Conv1d(16, 16, 7, 1, dilation=5, padding=15)))
    check("Conv1d, weight norm, dilation 5", m(x), nn.Conv1d(sd(m, "c"), "c", padding=15, dilation=5)(mx.array(x.numpy())))
    m = randomize(tnn.Conv1d(16, 8, 12, stride=6, padding=3))
    check("Conv1d, plain, k12 s6 p3", m(x), nn.Conv1d(sd(m, "c"), "c", stride=6, padding=3)(mx.array(x.numpy())))
    m = randomize(weight_norm(tnn.Conv1d(16, 8, 1, 1, 0, bias=False)))
    check("Conv1d, weight norm, 1x1 no bias", m(x), nn.Conv1d(sd(m, "c"), "c")(mx.array(x.numpy())))
    x1 = torch.randn(2, 1, 50)
    m = randomize(weight_norm(tnn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)))
    check("Conv1d, weight norm, 1 channel s2", m(x1), nn.Conv1d(sd(m, "c"), "c", stride=2, padding=1)(mx.array(x1.numpy())))

    m = randomize(weight_norm(tnn.ConvTranspose1d(16, 8, 20, 10, padding=5)))
    check("ConvTranspose1d, weight norm, k20 s10 p5", m(x), nn.ConvTranspose1d(sd(m, "c"), "c", stride=10, padding=5)(mx.array(x.numpy())))
    m = randomize(weight_norm(tnn.ConvTranspose1d(16, 8, 12, 6, padding=3)))
    check("ConvTranspose1d, weight norm, k12 s6 p3", m(x), nn.ConvTranspose1d(sd(m, "c"), "c", stride=6, padding=3)(mx.array(x.numpy())))
    m = randomize(weight_norm(tnn.ConvTranspose1d(16, 16, kernel_size=3, stride=2, groups=16, padding=1, output_padding=1)))
    check("ConvTranspose1d, depthwise, output_padding 1", m(x),
          nn.ConvTranspose1d(sd(m, "c"), "c", stride=2, padding=1, output_padding=1, groups=16)(mx.array(x.numpy())))

    xs = torch.randn(1, 37, 24)
    m = randomize(tnn.LSTM(24, 16, 1, batch_first=True, bidirectional=True))
    check("LSTM, bidirectional, 37 steps", m(xs)[0], nn.LSTM(sd(m, "l"), "l")(mx.array(xs.numpy())))
    xl = torch.randn(1, 700, 24)
    check("LSTM, bidirectional, 700 steps", m(xl)[0], nn.LSTM(sd(m, "l"), "l")(mx.array(xl.numpy())), tol=1e-4)

    g, b = torch.randn(24), torch.randn(24)
    check("layer_norm with weight and bias", F.layer_norm(xs, (24,), g, b, 1e-5),
          nn.layer_norm(mx.array(xs.numpy()), mx.array(g.numpy()), mx.array(b.numpy()), 1e-5))
    check("layer_norm, eps 1e-12", F.layer_norm(xs, (24,), g, b, 1e-12),
          nn.layer_norm(mx.array(xs.numpy()), mx.array(g.numpy()), mx.array(b.numpy()), 1e-12))
    check("instance_norm", tnn.InstanceNorm1d(16, affine=True)(x), nn.instance_norm(mx.array(x.numpy())))
    check("leaky_relu 0.2", F.leaky_relu(x, 0.2), nn.leaky_relu(mx.array(x.numpy()), 0.2))

    for n in (126, 245, 490, 1000, 2248):       # 126, 245, 490 and 1000 are lengths where rounding up would add a frame
        r = torch.rand(1, 9, 300 * n)
        down = F.interpolate(r, scale_factor=1 / 300, mode="linear")
        ours = nn.interpolate_linear(mx.array(r.numpy()), 1 / 300)
        check(f"interpolate linear x1/300, {n} frames", down, ours)
        up = F.interpolate(down * 300, scale_factor=300, mode="linear")
        check(f"interpolate linear x300, {n} frames", up, nn.interpolate_linear(ours * 300, 300))
    f0 = torch.rand(1, 1, 2248) * 300
    check("upsample nearest x300", tnn.Upsample(scale_factor=300)(f0), nn.upsample_nearest(mx.array(f0.numpy()), 300), tol=0)
    check("upsample nearest x2", F.interpolate(x, scale_factor=2, mode="nearest"), nn.upsample_nearest(mx.array(x.numpy()), 2), tol=0)

print("GATE 1 layers:", "PASS" if not fails else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
