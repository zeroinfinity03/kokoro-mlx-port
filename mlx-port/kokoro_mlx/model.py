"""KModel: phonemes and a voice style in, audio and the length of every phoneme out.

repo/kokoro/model.py, statement for statement. Like the official KModel it knows nothing about
languages: text becomes phonemes in KPipeline (pipeline.py).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from .albert import Albert
from .istftnet import Decoder
from .modules import ProsodyPredictor, TextEncoder
from .nn import linear

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SAMPLE_RATE = 24000


class KModel:
    def __init__(self, models_dir=None):
        """Loads config.json and kokoro-v1_0.safetensors (written by export/export_weights.py: the official
        tensors under their official names) from mlx-port/models, or from `models_dir`."""
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        config = json.loads((self.models_dir / "config.json").read_text(encoding="utf-8"))
        self.vocab = config["vocab"]
        w = self.weights = mx.load(str(self.models_dir / "kokoro-v1_0.safetensors"))
        assert all(v.dtype == mx.float32 for v in w.values()), "the model runs in float32, like the official one"
        self.bert = Albert(w, config["plbert"])
        self.context_length = config["plbert"]["max_position_embeddings"]
        self.predictor = ProsodyPredictor(w, style_dim=config["style_dim"], d_hid=config["hidden_dim"], nlayers=config["n_layer"])
        self.text_encoder = TextEncoder(w, kernel_size=config["text_encoder_kernel_size"], depth=config["n_layer"])
        self.decoder = Decoder(w, dim_in=config["hidden_dim"], **config["istftnet"])

    @dataclass
    class Output:
        audio: np.ndarray                       # float32, one row of samples at 24,000 Hz
        pred_dur: Optional[np.ndarray] = None   # frames (of 600 samples) for <bos>, each phoneme, <eos>

    def forward_with_tokens(self, input_ids, ref_s, speed=1, draws=None, trace=None):
        """model.py 86-119. input_ids [1, T] (0, the phoneme ids, 0); ref_s [1, 256], the voice's style row.
        `draws` replays recorded random numbers and `trace` collects the intermediate arrays (both for tests)."""
        t = trace if trace is not None else {}
        input_ids = mx.array(input_ids)
        ref_s = mx.array(ref_s)
        assert input_ids.shape[0] == 1, "one utterance at a time, as the official pipeline runs it"
        t["bert_dur"] = bert_dur = self.bert(input_ids)
        t["d_en"] = d_en = linear(bert_dur, self.weights, "bert_encoder").transpose(0, 2, 1)
        s = ref_s[:, 128:]
        t["d"] = d = self.predictor.text_encoder(d_en, s)
        t["lstm_x"] = x = self.predictor.lstm(d)
        t["duration_logits"] = duration = self.predictor.duration_proj(x)
        t["duration"] = duration = mx.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = np.array(mx.maximum(mx.round(duration), 1).astype(mx.int32))[0]      # round half to even, like torch.round
        t["pred_dur"] = pred_dur
        # the alignment matrix of the official code is 0/1, one 1 per output frame; multiplying by it copies
        # column `i` of the features pred_dur[i] times, which is done here directly
        indices = mx.array(np.repeat(np.arange(input_ids.shape[1]), pred_dur))
        t["en"] = en = mx.take(d.transpose(0, 2, 1), indices, axis=2)
        t["F0_pred"], t["N_pred"] = F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t["t_en"] = t_en = self.text_encoder(input_ids)
        t["asr"] = asr = mx.take(t_en, indices, axis=2)
        t["gen_x"] = gen_x = self.decoder.features(asr, F0_pred, N_pred, ref_s[:, :128])
        audio = self.decoder.generator(gen_x, ref_s[:, :128], F0_pred, draws)[0, 0]
        mx.eval(audio)
        return np.array(audio), pred_dur

    def __call__(self, phonemes, ref_s, speed=1, return_output=False):
        """model.py 121-137: phonemes not in the vocabulary are dropped, 0 is added at both ends."""
        input_ids = [i for i in (self.vocab.get(p) for p in phonemes) if i is not None]
        assert len(input_ids) + 2 <= self.context_length, (len(input_ids) + 2, self.context_length)
        audio, pred_dur = self.forward_with_tokens(np.array([[0, *input_ids, 0]]), ref_s, speed)
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio
