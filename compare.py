import sys
import time
from pathlib import Path

import numpy as np
import soundfile
from mlx_audio.tts.utils import load_model

here = Path(__file__).parent
reference = str(here / "hikari_voice.wav")
reference_text = (here / "hikari_voice.txt").read_text()
text = "Hey Kevin, I'm here. It's a little chilly out today, so maybe grab a jacket before you head out. Want me to set a reminder?"

for model_name in sys.argv[1:]:
    start = time.perf_counter()
    model = load_model(model_name)
    loaded = time.perf_counter() - start
    for attempt in ("warm", "timed"):
        start = time.perf_counter()
        chunks = [np.array(result.audio) for result in model.generate(text=text, ref_audio=reference, ref_text=reference_text)]
        elapsed = time.perf_counter() - start
    audio = np.concatenate(chunks)
    seconds = len(audio) / model.sample_rate
    name = model_name.split("/")[-1]
    soundfile.write(here / f"clone_{name}.wav", audio, model.sample_rate)
    print(f"{name}: load {loaded:.1f}s, {elapsed:.2f}s for {seconds:.1f}s audio, rtf {elapsed / seconds:.2f}, rate {model.sample_rate}", flush=True)
    del model
