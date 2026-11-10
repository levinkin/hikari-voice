import queue
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from mlx_audio.tts.generate import load_audio
from mlx_audio.tts.utils import load_model
from pydantic import BaseModel

here = Path(__file__).parent
mx.set_cache_limit(0)
model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
reference_audio = load_audio(str(here / "hikari_voice.wav"), sample_rate=model.sample_rate)
reference_text = (here / "hikari_voice.txt").read_text()
list(model.generate(text="Hello there, Kevin.", ref_audio=reference_audio, ref_text=reference_text, stream=True, streaming_interval=0.5))
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


def synthesize(text: str, chunks: queue.Queue, stop: threading.Event):
    try:
        with lock:
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
    threading.Thread(target=synthesize, args=(request.input, chunks, stop), daemon=True).start()
    return StreamingResponse(stream(chunks, stop), media_type="audio/pcm")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
