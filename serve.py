import platform
import queue
import re
import sys
import threading
import tomllib
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

here = Path(__file__).parent
settings = tomllib.loads((here / "settings.toml").read_text())
apple = platform.system() == "Darwin" and platform.machine() == "arm64"
if apple:
    import mlx_backend as backend
else:
    import torch
    if not torch.cuda.is_available():
        sys.exit("hikari-voice needs an apple silicon mac or an nvidia/amd gpu")
    import torch_backend as backend
model_name = settings.get("model", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit" if apple else "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
voice = backend.load(model_name, here / settings["voice"], settings["transcript"])
lock = threading.Lock()
app = FastAPI()


def keep_warm():
    while not threading.Event().wait(120):
        if lock.acquire(blocking=False):
            try:
                list(backend.speak(voice, "Hi.", None))
            finally:
                lock.release()


threading.Thread(target=keep_warm, daemon=True).start()


class SpeechRequest(BaseModel):
    input: str
    seed: int | None = None


def synthesize(text: str, seed: int | None, chunks: queue.Queue, stop: threading.Event):
    try:
        with lock:
            for audio in backend.speak(voice, text, seed):
                if stop.is_set():
                    break
                chunks.put((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
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
