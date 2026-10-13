"""KPipeline: text in, speech and word times out.

repo/kokoro/pipeline.py, statement for statement: the same text-to-phoneme package (misaki), the same
chunking, the same voice style row, the same word-time arithmetic. What differs from the official class:
the model and the voices are read from mlx-port/models (nothing is downloaded), `result.audio` and
`result.pred_dur` are numpy arrays, and `repo_id` / `device` are accepted and ignored so that existing
calls keep working.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from misaki import en, espeak

from .model import MODELS_DIR, KModel

ALIASES = {
    'en-us': 'a',
    'en-gb': 'b',
    'es': 'e',
    'fr-fr': 'f',
    'hi': 'h',
    'it': 'i',
    'pt-br': 'p',
    'ja': 'j',
    'zh': 'z',
}

LANG_CODES = dict(
    # pip install misaki[en]
    a='American English',
    b='British English',

    # espeak-ng
    e='es',
    f='fr-fr',
    h='hi',
    i='it',
    p='pt-br',

    # pip install misaki[ja]
    j='Japanese',

    # pip install misaki[zh]
    z='Mandarin Chinese',
)


class KPipeline:
    '''
    One KPipeline per language; several pipelines can share one KModel:
        model = KModel(); us = KPipeline('a', model=model); gb = KPipeline('b', model=model)

    A pipeline built with model=False makes no audio: it yields (graphemes, phonemes, None), to turn text
    into phonemes and chunks in advance.
    '''
    def __init__(
        self,
        lang_code: str,
        repo_id: Optional[str] = None,
        model: Union[KModel, bool] = True,
        trf: bool = False,
        en_callable: Optional[Callable[[str], str]] = None,
        device: Optional[str] = None,
        models_dir=None,
    ):
        lang_code = lang_code.lower()
        lang_code = ALIASES.get(lang_code, lang_code)
        assert lang_code in LANG_CODES, (lang_code, LANG_CODES)
        self.lang_code = lang_code
        self.model = None
        if isinstance(model, KModel):
            self.model = model
        elif model:
            self.model = KModel(models_dir)
        self.models_dir = self.model.models_dir if self.model else models_dir
        self._packs = None
        self.voices = {}
        if lang_code in 'ab':
            try:
                fallback = espeak.EspeakFallback(british=lang_code=='b')
            except Exception as e:
                print(f"kokoro_mlx: EspeakFallback not enabled, out-of-dictionary words will be skipped ({e})")
                fallback = None
            self.g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback, unk='')
        elif lang_code == 'j':
            from misaki import ja          # pip install misaki[ja]
            self.g2p = ja.JAG2P()
        elif lang_code == 'z':
            from misaki import zh          # pip install misaki[zh]
            # the official call picks version '1.1' only for the v1.1-zh model; these are the Kokoro-82M weights
            self.g2p = zh.ZHG2P(version=None, en_callable=en_callable)
        else:
            language = LANG_CODES[lang_code]
            # chunking is not implemented for these languages (officially too): split long texts with '\n'
            self.g2p = espeak.EspeakG2P(language=language)

    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if self._packs is None:
            self._packs = mx.load(str(Path(self.models_dir or MODELS_DIR) / "voices.safetensors"))
        if voice not in self._packs:
            raise ValueError(f"No voice named {voice!r}. Voices: {', '.join(sorted(self._packs))}")
        pack = self._packs[voice]            # [510, 1, 256]: one style row per phoneme count
        self.voices[voice] = pack
        return pack

    def load_voice(self, voice: Union[str, mx.array], delimiter: str = ","):
        """One voice ('af_bella') or several ('af_bella,af_jessica'), which are averaged."""
        if isinstance(voice, mx.array):
            return voice
        if voice in self.voices:
            return self.voices[voice]
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = mx.mean(mx.stack(packs), axis=0)
        return self.voices[voice]

    @staticmethod
    def tokens_to_ps(tokens: List[en.MToken]) -> str:
        return ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()

    @staticmethod
    def waterfall_last(
        tokens: List[en.MToken],
        next_count: int,
        waterfall: List[str] = ['!.?…', ':;', ',—'],
        bumps: List[str] = [')', '”']
    ) -> int:
        for w in waterfall:
            z = next((i for i, t in reversed(list(enumerate(tokens))) if t.phonemes in set(w)), None)
            if z is None:
                continue
            z += 1
            if z < len(tokens) and tokens[z].phonemes in bumps:
                z += 1
            if next_count - len(KPipeline.tokens_to_ps(tokens[:z])) <= 510:
                return z
        return len(tokens)

    @staticmethod
    def tokens_to_text(tokens: List[en.MToken]) -> str:
        return ''.join(t.text + t.whitespace for t in tokens).strip()

    def en_tokenize(
        self,
        tokens: List[en.MToken]
    ) -> Generator[Tuple[str, str, List[en.MToken]], None, None]:
        tks = []
        pcount = 0
        for t in tokens:
            # the official line leaves phonemes as misaki gives them (its `ɾ` -> `T` replace is commented out)
            t.phonemes = '' if t.phonemes is None else t.phonemes
            next_ps = t.phonemes + (' ' if t.whitespace else '')
            next_pcount = pcount + len(next_ps.rstrip())
            if next_pcount > 510:
                z = KPipeline.waterfall_last(tks, next_pcount)
                text = KPipeline.tokens_to_text(tks[:z])
                ps = KPipeline.tokens_to_ps(tks[:z])
                yield text, ps, tks[:z]
                tks = tks[z:]
                pcount = len(KPipeline.tokens_to_ps(tks))
                if not tks:
                    next_ps = next_ps.lstrip()
            tks.append(t)
            pcount += len(next_ps)
        if tks:
            text = KPipeline.tokens_to_text(tks)
            ps = KPipeline.tokens_to_ps(tks)
            yield ''.join(text).strip(), ''.join(ps).strip(), tks

    @staticmethod
    def infer(
        model: KModel,
        ps: str,
        pack,
        speed: Union[float, Callable[[int], float]] = 1
    ) -> KModel.Output:
        if callable(speed):
            speed = speed(len(ps))
        # the style row is chosen by the number of phonemes
        return model(ps, pack[len(ps)-1], speed, return_output=True)

    def generate_from_tokens(
        self,
        tokens: Union[str, List[en.MToken]],
        voice: str,
        speed: float = 1,
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        """Audio from a phoneme string, or from tokens misaki already made."""
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: pipeline.generate_from_tokens(..., voice="af_heart")')

        pack = self.load_voice(voice) if model else None

        # a raw phoneme string
        if isinstance(tokens, str):
            if len(tokens) > 510:
                raise ValueError(f'Phoneme string too long: {len(tokens)} > 510')
            output = KPipeline.infer(model, tokens, pack, speed) if model else None
            yield self.Result(graphemes='', phonemes=tokens, output=output)
            return

        for gs, ps, tks in self.en_tokenize(tokens):
            if not ps:
                continue
            elif len(ps) > 510:
                ps = ps[:510]
            output = KPipeline.infer(model, ps, pack, speed) if model else None
            if output is not None and output.pred_dur is not None:
                KPipeline.join_timestamps(tks, output.pred_dur)
            yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output)

    @staticmethod
    def join_timestamps(tokens: List[en.MToken], pred_dur):
        # Multiply by 600 to go from pred_dur frames to sample_rate 24000
        # Equivalent to dividing pred_dur frames by 40 to get timestamp in seconds
        # We will count nice round half-frames, so the divisor is 80
        MAGIC_DIVISOR = 80
        if not tokens or len(pred_dur) < 3:
            # We expect at least 3: <bos>, token, <eos>
            return
        # We track 2 counts, measured in half-frames: (left, right)
        # This way we can cut space characters in half
        left = right = 2 * max(0, pred_dur[0].item() - 3)
        # Updates:
        # left = right + (2 * token_dur) + space_dur
        # right = left + space_dur
        i = 1
        for t in tokens:
            if i >= len(pred_dur)-1:
                break
            if not t.phonemes:
                if t.whitespace:
                    i += 1
                    left = right + pred_dur[i].item()
                    right = left + pred_dur[i].item()
                    i += 1
                continue
            j = i + len(t.phonemes)
            if j >= len(pred_dur):
                break
            t.start_ts = left / MAGIC_DIVISOR
            token_dur = pred_dur[i: j].sum().item()
            space_dur = pred_dur[j].item() if t.whitespace else 0
            left = right + (2 * token_dur) + space_dur
            t.end_ts = left / MAGIC_DIVISOR
            right = left + space_dur
            i = j + (1 if t.whitespace else 0)

    @dataclass
    class Result:
        graphemes: str
        phonemes: str
        tokens: Optional[List[en.MToken]] = None
        output: Optional[KModel.Output] = None
        text_index: Optional[int] = None

        @property
        def audio(self) -> Optional[np.ndarray]:
            return None if self.output is None else self.output.audio

        @property
        def pred_dur(self) -> Optional[np.ndarray]:
            return None if self.output is None else self.output.pred_dur

        # the official Result also unpacks as (graphemes, phonemes, audio)
        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index):
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self):
            return 3

    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Union[float, Callable[[int], float]] = 1,
        split_pattern: Optional[str] = r'\n+',
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: en_us_pipeline(text="Hello world!", voice="af_heart")')
        pack = self.load_voice(voice) if model else None

        # Convert input to list of segments
        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]

        # Process each segment
        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():  # Skip empty segments
                continue

            # English
            if self.lang_code in 'ab':
                _, tokens = self.g2p(graphemes)
                for gs, ps, tks in self.en_tokenize(tokens):
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        ps = ps[:510]
                    output = KPipeline.infer(model, ps, pack, speed) if model else None
                    if output is not None and output.pred_dur is not None:
                        KPipeline.join_timestamps(tks, output.pred_dur)
                    yield self.Result(graphemes=gs, phonemes=ps, tokens=tks, output=output, text_index=graphemes_index)

            # Other languages: chunks of roughly 400 characters, cut at sentence ends when possible
            else:
                chunk_size = 400
                chunks = []

                sentences = re.split(r'([.!?]+)', graphemes)
                current_chunk = ""

                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    # Add the punctuation back if it exists
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]

                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk.strip())

                # no sentence ends at all: cut by length
                if not chunks:
                    chunks = [graphemes[i:i+chunk_size] for i in range(0, len(graphemes), chunk_size)]

                for chunk in chunks:
                    if not chunk.strip():
                        continue

                    ps, _ = self.g2p(chunk)
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        ps = ps[:510]

                    output = KPipeline.infer(model, ps, pack, speed) if model else None
                    yield self.Result(graphemes=chunk, phonemes=ps, output=output, text_index=graphemes_index)
