"""The phoneme-level ALBERT (PL-BERT) Kokoro uses to plan durations and pitch.

Official: `CustomAlbert` (repo/kokoro/modules.py 180-183) is Hugging Face's `AlbertModel`
(repo/transformers_modeling_albert.py) returning `last_hidden_state`. Its config is
`AlbertConfig(vocab_size=n_token, **config["plbert"])` (repo/kokoro/model.py 51), so everything the
plbert section does not name keeps Hugging Face's default: embedding_size 128, one shared layer
(num_hidden_groups 1, inner_group_num 1), layer_norm_eps 1e-12, and hidden_act "gelu_new".
"""
import math

import mlx.core as mx

from .nn import layer_norm, linear

EPS = 1e-12


def gelu_new(x):
    """Hugging Face's "gelu_new", ALBERT's default activation: the tanh form of GELU
    (transformers/activations.py NewGELUActivation). It is not the exact erf GELU; the two differ by up
    to 5e-4, enough to move a predicted duration across a rounding boundary."""
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


class Albert:
    def __init__(self, w, plbert, prefix="bert"):
        self.w, self.p = w, prefix
        self.heads = plbert["num_attention_heads"]
        self.layers = plbert["num_hidden_layers"]
        self.hidden = plbert["hidden_size"]
        self.layer = f"{prefix}.encoder.albert_layer_groups.0.albert_layers.0"

    def __call__(self, input_ids):
        """input_ids (B, T), every position real (the official call passes an all-ones attention mask,
        which adds nothing to the scores). Returns last_hidden_state (B, T, hidden)."""
        w, p, L = self.w, self.p, self.layer
        B, T = input_ids.shape
        # AlbertEmbeddings: word + token type 0 + position, LayerNorm (dropout is off at inference)
        x = (w[f"{p}.embeddings.word_embeddings.weight"][input_ids]
             + w[f"{p}.embeddings.token_type_embeddings.weight"][0]
             + w[f"{p}.embeddings.position_embeddings.weight"][:T])
        x = layer_norm(x, w[f"{p}.embeddings.LayerNorm.weight"], w[f"{p}.embeddings.LayerNorm.bias"], EPS)
        x = linear(x, w, f"{p}.encoder.embedding_hidden_mapping_in")
        H, D = self.heads, self.hidden // self.heads
        for _ in range(self.layers):          # the same layer, applied num_hidden_layers times
            # AlbertAttention
            q = linear(x, w, f"{L}.attention.query").reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = linear(x, w, f"{L}.attention.key").reshape(B, T, H, D).transpose(0, 2, 1, 3)
            v = linear(x, w, f"{L}.attention.value").reshape(B, T, H, D).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D)
            ctx = (mx.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, T, self.hidden)
            attn = layer_norm(x + linear(ctx, w, f"{L}.attention.dense"),
                              w[f"{L}.attention.LayerNorm.weight"], w[f"{L}.attention.LayerNorm.bias"], EPS)
            # AlbertLayer.ff_chunk, then the residual LayerNorm
            ffn = linear(gelu_new(linear(attn, w, f"{L}.ffn")), w, f"{L}.ffn_output")
            x = layer_norm(ffn + attn, w[f"{L}.full_layer_layer_norm.weight"], w[f"{L}.full_layer_layer_norm.bias"], EPS)
        return x
