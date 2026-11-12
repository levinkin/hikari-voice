import re

import librosa
import numpy as np

import serve

LINES = [
    "Hey Kevin, I'm here. Want me to set a reminder?",
    "Nya, that tickles! Hehe.",
    "Want me to check the weather for tomorrow?",
    "Oh no, I'm really sorry. That sounds like a rough day.",
    "Your meeting starts in ten minutes.",
    "Wow, you actually finished it! That's amazing!",
    "Did you remember to drink some water today?",
    "I put the grocery list in your notes.",
    "I miss you when you're gone all day. It gets so quiet here.",
    "Okay, first preheat the oven, then mix the flour and sugar, and after that we can talk about what to make for dinner tomorrow night.",
]
SEEDS = range(1, 31)
TEMPERATURES = [0.9, 0.8, 0.7, 0.6]


def pitch(audio):
    f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=600, sr=24000, frame_length=2048, hop_length=240)
    return np.where(voiced, f0, np.nan)


def score(f0, floor):
    low = np.concatenate([[0], f0 < floor, [0]]).astype(int)
    edges = np.flatnonzero(np.diff(low))
    voiced = f0[~np.isnan(f0)]
    if voiced.size == 0:
        return False, np.nan
    p10, p90 = np.percentile(voiced, [10, 90])
    return np.any(edges[1::2] - edges[::2] >= 10), 12 * np.log2(p90 / p10)


def save(temperature):
    path = serve.here / "settings.toml"
    text = path.read_bytes().decode()
    text, found = re.subn(r"^[ \t]*temperature[ \t]*=[^\r\n]*", f"temperature = {temperature}", text, count=1, flags=re.MULTILINE)
    if not found:
        newline = "\r\n" if "\r\n" in text else "\n"
        text = text + ("" if text.endswith("\n") or not text else newline) + f"temperature = {temperature}{newline}"
    path.write_bytes(text.encode())


if __name__ == "__main__":
    reference, _ = librosa.load(serve.here / serve.settings["voice"], sr=24000)
    floor = 0.6 * np.nanmedian(pitch(reference))
    print(f"reference median pitch {floor / 0.6:.0f} hz, growl below {floor:.0f} hz", flush=True)
    voice = serve.backend.load(serve.model_name, serve.here / serve.settings["voice"], serve.settings["transcript"], serve.temperature)
    results = {}
    for temperature in TEMPERATURES:
        growls, ranges = 0, []
        for number, line in enumerate(LINES, 1):
            for seed in SEEDS:
                audio = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in serve.backend.speak(voice, serve.prepare(line), seed, temperature)])
                growl, semitones = score(pitch(audio), floor)
                growls += growl
                ranges.append(semitones)
            print(f"  {temperature}: line {number}/{len(LINES)}, {growls} growls so far", flush=True)
        results[temperature] = growls
        print(f"temperature {temperature}: {growls}/{len(LINES) * len(SEEDS)} growls, pitch range {np.nanmean(ranges):.2f} semitones", flush=True)
    best = min(TEMPERATURES, key=results.get)
    save(best)
    print(f"wrote temperature = {best} to settings.toml")
