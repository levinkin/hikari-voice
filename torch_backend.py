from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS


class DiskEmbedding(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = table

    def forward(self, token_ids):
        rows = torch.from_numpy(self.table[token_ids.cpu().numpy().reshape(-1)]).view(torch.bfloat16)
        return rows.to(token_ids.device).reshape(*token_ids.shape, self.table.shape[1])


def load(model_name, voice_path, reference_text, temperature):
    table_path = Path(__file__).parent / f"text_table-{quote(model_name, safe='')}.u16"
    model = FasterQwen3TTS.from_pretrained(model_name)
    qwen = model.model.model
    text_weight = qwen.talker.model.text_embedding.weight
    if not table_path.exists():
        text_weight.view(torch.int16).cpu().numpy().tofile(table_path)
    qwen.talker.model.text_embedding = DiskEmbedding(np.memmap(table_path, dtype=np.int16, mode="r", shape=tuple(text_weight.shape)))
    del text_weight
    voice = (model, str(voice_path), reference_text)
    list(speak(voice, "Hello there.", None, temperature))
    qwen.speaker_encoder = None
    qwen.speech_tokenizer.model.encoder = None
    torch.cuda.empty_cache()
    return voice


def speak(voice, text, seed, temperature):
    model, voice_path, reference_text = voice
    if seed is not None:
        torch.manual_seed(seed)
    for audio, _, _ in model.generate_voice_clone_streaming(text, "Auto", ref_audio=voice_path, ref_text=reference_text, append_silence=False, chunk_size=8, temperature=temperature):
        yield audio
