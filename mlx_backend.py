import gc
import types
from pathlib import Path
from urllib.parse import quote

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_audio.tts.generate import load_audio
from mlx_audio.tts.utils import load_model


class DiskEmbedding(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def __call__(self, token_ids):
        ids = np.array(token_ids)
        rows = mx.array(self.table[ids.reshape(-1)]).view(mx.bfloat16)
        return rows.reshape(*ids.shape, self.table.shape[1])


def cached_speaker_embedding(self, audio, sr=24000):
    key = (audio.size, float(audio.sum()))
    if key not in self.speaker_cache:
        self.speaker_cache[key] = self.original_speaker_embedding(audio, sr)
        mx.eval(self.speaker_cache[key])
    return self.speaker_cache[key]


def load(model_name, voice_path, reference_text):
    table_path = Path(__file__).parent / f"text_table-{quote(model_name, safe='')}.u16"
    mx.set_cache_limit(0)
    model = load_model(model_name)
    reference_audio = load_audio(str(voice_path), sample_rate=model.sample_rate)
    model.speaker_cache = {}
    model.original_speaker_embedding = model.extract_speaker_embedding
    model.extract_speaker_embedding = types.MethodType(cached_speaker_embedding, model)
    text_weight = model.talker.model.text_embedding.weight
    if not table_path.exists():
        np.array(text_weight.view(mx.uint16)).tofile(table_path)
    model.talker.model.text_embedding = DiskEmbedding(np.memmap(table_path, dtype=np.uint16, mode="r", shape=text_weight.shape))
    del text_weight
    voice = (model, reference_audio, reference_text)
    list(speak(voice, "Hello there.", None))
    model.speech_tokenizer.encoder_model = types.SimpleNamespace()
    model.speaker_encoder = types.SimpleNamespace()
    gc.collect()
    mx.clear_cache()
    return voice


def speak(voice, text, seed):
    model, reference_audio, reference_text = voice
    if seed is not None:
        mx.random.seed(seed)
    for result in model.generate(text=text, ref_audio=reference_audio, ref_text=reference_text, stream=True, streaming_interval=0.5, temperature=0.7):
        yield np.array(result.audio)
