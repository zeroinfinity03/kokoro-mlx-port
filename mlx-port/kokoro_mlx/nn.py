"""Building blocks: the PyTorch layers Kokoro uses, written for MLX.

Every layer reads its tensors from the weight dict by their official names and takes and returns
arrays in PyTorch's layout (batch, channels, time), so the model code above reads like the official
code. MLX convolutions want (batch, time, channels) and kernels as (out, k, in): the swap happens
inside each layer, in one place.
"""
import math

import mlx.core as mx
import numpy as np


def linear(x, w, name):
    """torch.nn.Linear: weight is (out, in)."""
    y = x @ w[name + ".weight"].T
    b = w.get(name + ".bias")
    return y if b is None else y + b


def layer_norm(x, weight=None, bias=None, eps=1e-5):
    """F.layer_norm over the last axis: biased variance, eps inside the square root."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / mx.sqrt(var + eps)
    if weight is not None:
        y = y * weight + bias
    return y


def instance_norm(x, eps=1e-5):
    """nn.InstanceNorm1d on (B, C, T): each channel of each sample over time, biased variance.
    Kokoro builds it with affine=True, but the checkpoint has no weight or bias for it and the official
    loader is not strict, so they stay 1 and 0 (repo/kokoro/istftnet.py 23-24, model.py 68-75)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


def leaky_relu(x, slope):
    return mx.where(x >= 0, x, x * slope)


def _weight(w, name):
    """A layer's weight in PyTorch layout. Layers wrapped in torch.nn.utils.weight_norm store weight_g and
    weight_v instead: weight = v * (g / ||v||), the norm over every axis but the first (dim=0). For a
    ConvTranspose1d the first axis is the INPUT channel; that is what torch does and what is done here."""
    if name + ".weight_g" in w:
        g, v = w[name + ".weight_g"], w[name + ".weight_v"]
        norm = mx.sqrt((v * v).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
        return v * (g / norm)
    return w[name + ".weight"]


class Conv1d:
    """torch.nn.Conv1d. PyTorch weight (out, in/groups, k) -> MLX (out, k, in/groups)."""

    def __init__(self, w, name, stride=1, padding=0, dilation=1, groups=1):
        self.weight = _weight(w, name).transpose(0, 2, 1)
        self.bias = w.get(name + ".bias")
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups

    def __call__(self, x):
        y = mx.conv1d(x.transpose(0, 2, 1), self.weight, stride=self.stride, padding=self.padding,
                      dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(0, 2, 1)


class ConvTranspose1d:
    """torch.nn.ConvTranspose1d. PyTorch weight (in, out/groups, k) -> MLX (out, k, in/groups).
    Kokoro uses two kinds: groups=1 (the generator's upsamplers) and depthwise, groups == channels
    (the `pool` of an upsampling AdainResBlk1d)."""

    def __init__(self, w, name, stride=1, padding=0, output_padding=0, groups=1):
        wt = _weight(w, name)
        if groups == 1:
            self.weight = wt.transpose(1, 2, 0)
        else:
            assert groups == wt.shape[0] and wt.shape[1] == 1, "only depthwise grouped transposed convs"
            self.weight = wt.transpose(0, 2, 1)
        self.bias = w.get(name + ".bias")
        self.stride, self.padding, self.output_padding, self.groups = stride, padding, output_padding, groups

    def __call__(self, x):
        y = mx.conv_transpose1d(x.transpose(0, 2, 1), self.weight, stride=self.stride, padding=self.padding,
                                output_padding=self.output_padding, groups=self.groups)
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(0, 2, 1)


class LSTM:
    """torch.nn.LSTM(num_layers=1, batch_first=True, bidirectional=True) on (B, T, in) -> (B, T, 2*hidden).
    Gate order in the weights is i, f, g, o; both biases are added; initial state is zero. The backward
    direction walks the sequence from the end and its outputs stay at their own time steps."""

    def __init__(self, w, name):
        self.dirs = [tuple(w[f"{name}.{k}_l0{sfx}"] for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
                     for sfx in ("", "_reverse")]

    @staticmethod
    def _run(x, w_ih, w_hh, b_ih, b_hh, reverse):
        B, T, _ = x.shape
        H = w_hh.shape[1]
        xi = x @ w_ih.T + b_ih
        h = mx.zeros((B, H), dtype=x.dtype)
        c = mx.zeros((B, H), dtype=x.dtype)
        out = [None] * T
        for t in (range(T - 1, -1, -1) if reverse else range(T)):
            gates = xi[:, t] + (h @ w_hh.T + b_hh)
            i, f, g, o = mx.split(gates, 4, axis=-1)
            c = mx.sigmoid(f) * c + mx.sigmoid(i) * mx.tanh(g)
            h = mx.sigmoid(o) * mx.tanh(c)
            out[t] = h
        return mx.stack(out, axis=1)

    def __call__(self, x):
        return mx.concatenate([self._run(x, *self.dirs[0], reverse=False), self._run(x, *self.dirs[1], reverse=True)], axis=-1)


def interpolate_linear(x, scale_factor):
    """F.interpolate(x, scale_factor=scale_factor, mode="linear") on (B, C, L), align_corners=False.
    As PyTorch does it (aten/src/ATen/native/UpSample.h): the output length is floor(L * scale_factor),
    rounded DOWN in double precision, and each output sample reads the input at
    ratio * (dst + 0.5) - 0.5, never below 0, with ratio = float32(1 / scale_factor), blended between
    its two neighbours. The position is worked out here on the CPU as an exact product rounded once to
    float32, which is what PyTorch's fused multiply-subtract gives; rounding the product and the
    subtraction separately lands one float32 step away for some samples, and at 28 s of audio that
    step is a visible error in the phase."""
    L = x.shape[-1]
    out = int(math.floor(float(L) * float(scale_factor)))
    ratio = np.float64(np.float32(1.0 / float(scale_factor)))
    src = np.maximum((ratio * (np.arange(out, dtype=np.float64) + 0.5) - 0.5).astype(np.float32), np.float32(0))
    i0 = np.minimum(src.astype(np.int64), L - 1)
    i1 = i0 + (i0 < L - 1)
    lam1 = np.clip(src - i0.astype(np.float32), 0, 1).astype(np.float32)
    lam0 = (np.float32(1) - lam1).astype(np.float32)
    return mx.array(lam0) * mx.take(x, mx.array(i0), axis=-1) + mx.array(lam1) * mx.take(x, mx.array(i1), axis=-1)


def upsample_nearest(x, factor):
    """F.interpolate(mode="nearest") / nn.Upsample by a whole number on (B, C, L): every sample repeated."""
    return mx.repeat(x, factor, axis=-1)
