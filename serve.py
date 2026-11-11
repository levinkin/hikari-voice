import gc
import queue
import re
import threading
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from mlx_audio.tts.generate import load_audio
from mlx_audio.tts.utils import load_model
from pydantic import BaseModel


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


here = Path(__file__).parent
table_path = here / "text_table.u16"
mx.set_cache_limit(0)
model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
reference_audio = load_audio(str(here / "hikari_voice.wav"), sample_rate=model.sample_rate)
reference_text = (here / "hikari_voice.txt").read_text()
model.speaker_cache = {}
model.original_speaker_embedding = model.extract_speaker_embedding
model.extract_speaker_embedding = types.MethodType(cached_speaker_embedding, model)
text_weight = model.talker.model.text_embedding.weight
if not table_path.exists():
    np.array(text_weight.view(mx.uint16)).tofile(table_path)
model.talker.model.text_embedding = DiskEmbedding(np.memmap(table_path, dtype=np.uint16, mode="r", shape=text_weight.shape))
del text_weight
list(model.generate(text="Hello there.", ref_audio=reference_audio, ref_text=reference_text, stream=True, streaming_interval=0.5))
model.speech_tokenizer.encoder_model = types.SimpleNamespace()
model.speaker_encoder = types.SimpleNamespace()
gc.collect()
mx.clear_cache()
lock = threading.Lock()
app = FastAPI()


def keep_warm():
    while not threading.Event().wait(120):
        if lock.acquire(blocking=False):
            try:
                list(model.generate(text="Hi.", ref_audio=reference_audio, ref_text=reference_text, stream=True, streaming_interval=0.5))
            finally:
                lock.release()


threading.Thread(target=keep_warm, daemon=True).start()


class SpeechRequest(BaseModel):
    input: str
    seed: int | None = None


def synthesize(text: str, seed: int | None, chunks: queue.Queue, stop: threading.Event):
    try:
        with lock:
            if seed is not None:
                mx.random.seed(seed)
            for result in model.generate(text=text, ref_audio=reference_audio, ref_text=reference_text, stream=True, streaming_interval=0.5):
                if stop.is_set():
                    break
                chunks.put((np.clip(np.array(result.audio), -1, 1) * 32767).astype("<i2").tobytes())
    finally:
        chunks.put(None)


async def stream(chunks: queue.Queue, stop: threading.Event):
    try:
        while (chunk := await run_in_threadpool(chunks.get)) is not None:
            yield chunk
    finally:
        stop.set()


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest):
    chunks = queue.Queue()
    stop = threading.Event()
    text = re.sub(r"^\s*nya+\b[,~]?", "N'yah!", request.input, flags=re.IGNORECASE)
    text = re.sub(r"\bnya+\b", "N'yah", text, flags=re.IGNORECASE) + " ^."
    threading.Thread(target=synthesize, args=(text, request.seed, chunks, stop), daemon=True).start()
    return StreamingResponse(stream(chunks, stop), media_type="audio/pcm")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
