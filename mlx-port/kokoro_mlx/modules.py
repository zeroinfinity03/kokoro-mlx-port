"""Text encoder and prosody predictor: repo/kokoro/modules.py, statement for statement.

Arrays keep PyTorch's shapes, and each function names the official lines it follows. Kokoro runs one
utterance at a time at its full length (repo/kokoro/model.py 93-101: every length equals the width, so
the mask is all False), so the official pack_padded_sequence / masked_fill_ calls change nothing and are
not repeated here.
"""
import mlx.core as mx

from .istftnet import AdainResBlk1d
from .nn import LSTM, Conv1d, layer_norm, leaky_relu, linear


class TextEncoder:
    """modules.py 35-69: embedding, `depth` x (weight-normed conv, LayerNorm over channels, LeakyReLU 0.2),
    then a bidirectional LSTM."""

    def __init__(self, w, kernel_size, depth, prefix="text_encoder"):
        self.w, self.p, self.depth = w, prefix, depth
        self.convs = [Conv1d(w, f"{prefix}.cnn.{i}.0", padding=(kernel_size - 1) // 2) for i in range(depth)]
        self.lstm = LSTM(w, f"{prefix}.lstm")

    def __call__(self, input_ids):
        w, p = self.w, self.p
        x = w[f"{p}.embedding.weight"][input_ids]            # [B, T, emb]
        x = x.transpose(0, 2, 1)                             # [B, emb, T]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            # LayerNorm (modules.py 21-32): over the channel axis, with gamma and beta
            x = layer_norm(x.transpose(0, 2, 1), w[f"{p}.cnn.{i}.1.gamma"], w[f"{p}.cnn.{i}.1.beta"], 1e-5).transpose(0, 2, 1)
            x = leaky_relu(x, 0.2)
        x = self.lstm(x.transpose(0, 2, 1))                  # [B, T, chn]
        return x.transpose(0, 2, 1)                          # [B, chn, T]


def ada_layer_norm(x, s, w, name, eps=1e-5):
    """AdaLayerNorm (modules.py 72-88) on x [B, T, C] with style s [B, sty]. The official transposes cancel
    in pairs; what is left is a plain layer norm over C, scaled and shifted by the style:
    fc(s) -> first half gamma, second half beta -> (1 + gamma) * norm(x) + beta."""
    h = linear(s, w, f"{name}.fc")                          # [B, 2C]
    C = h.shape[1] // 2
    gamma, beta = h[:, None, :C], h[:, None, C:]             # [B, 1, C]
    return (1 + gamma) * layer_norm(x, eps=eps) + beta


class DurationEncoder:
    """modules.py 137-176: `nlayers` x (bidirectional LSTM, AdaLayerNorm), the style appended to the
    features before the first LSTM and again after every AdaLayerNorm."""

    def __init__(self, w, nlayers, prefix="predictor.text_encoder"):
        self.w, self.p, self.nlayers = w, prefix, nlayers
        self.lstms = [LSTM(w, f"{prefix}.lstms.{2 * i}") for i in range(nlayers)]

    def __call__(self, x, style):
        """x [B, C, T] (d_en), style [B, sty]. Returns [B, T, C + sty]."""
        B, C, T = x.shape
        s = mx.broadcast_to(style[:, None, :], (B, T, style.shape[-1]))     # the style at every step
        x = mx.concatenate([x.transpose(0, 2, 1), s], axis=-1)              # [B, T, C + sty]
        for i in range(self.nlayers):
            x = self.lstms[i](x)                                            # [B, T, C]
            x = ada_layer_norm(x, style, self.w, f"{self.p}.lstms.{2 * i + 1}")
            x = mx.concatenate([x, s], axis=-1)                             # [B, T, C + sty]
        return x


class ProsodyPredictor:
    """modules.py 91-134. The official inference path (model.py 105-115) calls its parts one by one:
    text_encoder, lstm, duration_proj, then F0Ntrain."""

    def __init__(self, w, style_dim, d_hid, nlayers, prefix="predictor"):
        self.w, self.p = w, prefix
        self.text_encoder = DurationEncoder(w, nlayers, f"{prefix}.text_encoder")
        self.lstm = LSTM(w, f"{prefix}.lstm")
        self.shared = LSTM(w, f"{prefix}.shared")
        self.F0 = [AdainResBlk1d(w, f"{prefix}.F0.0", d_hid, d_hid),
                   AdainResBlk1d(w, f"{prefix}.F0.1", d_hid, d_hid // 2, upsample=True),
                   AdainResBlk1d(w, f"{prefix}.F0.2", d_hid // 2, d_hid // 2)]
        self.N = [AdainResBlk1d(w, f"{prefix}.N.0", d_hid, d_hid),
                  AdainResBlk1d(w, f"{prefix}.N.1", d_hid, d_hid // 2, upsample=True),
                  AdainResBlk1d(w, f"{prefix}.N.2", d_hid // 2, d_hid // 2)]
        self.F0_proj = Conv1d(w, f"{prefix}.F0_proj")
        self.N_proj = Conv1d(w, f"{prefix}.N_proj")

    def duration_proj(self, x):
        """LinearNorm (modules.py 11-18): one Linear, named linear_layer."""
        return linear(x, self.w, f"{self.p}.duration_proj.linear_layer")

    def F0Ntrain(self, x, s):
        """modules.py 124-134. x [B, C + sty, frames] -> pitch and energy curves, each [B, 2 * frames]."""
        x = self.shared(x.transpose(0, 2, 1))
        F0 = x.transpose(0, 2, 1)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(0, 2, 1)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0[:, 0, :], N[:, 0, :]
