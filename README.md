# hikari voice

a small speech server that clones a voice from a 15 second clip with qwen3-tts. it runs on apple silicon macs, and on nvidia or amd gpus. intel macs aren't supported. i run it as the voice of hikari, my stackchan robot. it speaks every qwen3-tts language (english, japanese, chinese, korean, german, french, russian, portuguese, spanish, italian) and streams audio as it goes.

i wrote it because the stock way of running qwen3-tts through mlx-audio used more memory and started speaking slower than i wanted on a mac mini that also runs everything else.

## what it does differently

on the mac, all of this keeps the audio byte-identical to plain mlx-audio with the same text, seed and temperature. i checked english, japanese and german lines before and after, on all three mac models below.

- the text embedding table is 622 mb, and a sentence only touches a few dozen of its rows. serve.py writes the table to its own `text_table-<model>.u16` file once and reads rows from that file on demand instead of keeping it in ram.
- mlx-audio recomputes the speaker embedding from the reference clip for every sentence. serve.py computes it once and reuses it.
- after the first sentence, the reference audio encoders aren't needed anymore (their output is cached), so serve.py drops them.
- qwen3-tts sometimes cuts off the last word. adding ` ^.` to the end of the text fixes most of it, and the model doesn't read it out. on my test set it took cut-off endings from 7 out of 24 lines to 1.
- at the default temperature of 0.9, every now and then a sentence ends an octave low, like a growl. serve.py samples at 0.7 instead. over 200 takes each, that took growls from 4 to 0 on a 4090 and from 2 to 0 on the mac, with the same speed. a longer 300-take run on the 4090 still caught one at 0.7, so they're rarer, not gone. her pitch moves a little less (about 0.4 semitones of range).

on a gpu, serve.py uses [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (cuda graphs), which turns the reference clip into a voice prompt once and reuses it, and streams in smaller chunks so the first audio comes out sooner. the audio there isn't byte-identical to plain faster-qwen3-tts, it's just the same voice.

same 4 lines and seeds for each row, plain library vs serve.py:

| machine | model | | first audio | speed (rtf) | memory | peak while talking |
|---|---|---|---|---|---|---|
| mac mini | 0.6b 8-bit | plain | 281 ms | 0.37 | 2223 mb | 4598 mb |
| | | this | 271 ms | 0.37 | 1255 mb | 1594 mb |
| mac mini | 0.6b 4-bit | plain | 285 ms | 0.31 | 1956 mb | 4325 mb |
| | | this | 240 ms | 0.29 | 988 mb | 2034 mb |
| mac mini | 1.7b 8-bit | plain | 371 ms | 0.49 | 3287 mb | 5701 mb |
| | | this | 373 ms | 0.50 | 2316 mb | 3070 mb |
| rtx 4090 | 0.6b | plain | 255 ms | 0.25 | 3748 mb vram | 4142 mb vram |
| | | this | 188 ms | 0.26 | 3920 mb vram | 4368 mb vram |
| rtx 4090 | 1.7b | plain | 279 ms | 0.28 | 5836 mb vram | 6154 mb vram |
| | | this | 206 ms | 0.29 | 5792 mb vram | 6240 mb vram |
| rtx 4090, windows | 0.6b | plain | 381 ms | 0.32 | 3838 mb vram | 4156 mb vram |
| | | this | 224 ms | 0.39 | 3944 mb vram | 4702 mb vram |
| rtx 4090, windows | 1.7b | plain | 396 ms | 0.35 | 5033 mb vram | 5493 mb vram |
| | | this | 228 ms | 0.43 | 5931 mb vram | 6689 mb vram |

rtf is seconds spent per second of audio, so lower is faster. the mac 1.7b rows are the average of 3 runs each, taking turns with plain, while my other server was using the same gpu. the other rows are one run each. on the 4090 under linux the process also uses about 3 gb of normal ram either way (about 2.3 gb on windows). the smaller chunks cost about 5% speed on linux and about 20% on windows, but it still talks well over 2x faster than real time and starts about 70 ms sooner on linux and about 160 ms sooner on windows. the extra vram is pytorch holding on to memory it reserved, not memory the model uses. rows marked just rtx 4090 are linux. windows doesn't report vram per process, so its vram numbers are total gpu use minus what the desktop used before the start, give or take 300 mb. on windows, plain starts about 120 ms later and runs about 25% slower than on linux. this server starts only 20-35 ms later there, but runs about 50% slower.


## running it

### apple silicon

needs an apple silicon mac and python 3.12.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python serve.py
```

the first start downloads `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` and writes `text_table-<model>.u16` (about 600 mb). that first run holds the table in memory too, so restart it once to get the lower number. each model writes its own table (622 mb for the 1.7b too), and you can delete the ones you don't use anymore.

### nvidia or amd (linux, windows)

needs python 3.12 and a gpu with about 5 gb free for the 0.6b model. on linux the same commands work:

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python serve.py
```

on windows:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python serve.py
```

on linux, pip pulls the cuda build of torch. for amd, install the rocm build of torch 2.15.0 from pytorch.org first, then run the pip line above. on windows, pip's default torch has no gpu support, which is why the windows steps install the cuda build first. i tested both on an nvidia rtx 4090: install took 81 s on linux and about 90 s on windows, and once the model is downloaded, startup takes about 7 s on linux and 11-12 s on windows. i haven't tried an amd card. the first start downloads `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

there's no cpu mode. without an apple silicon mac or a gpu, serve.py stops with `hikari-voice needs an apple silicon mac or an nvidia/amd gpu`. plain qwen-tts does run on cpu, but on a 32-thread desktop it took 13 s before any audio and 4.3 s per second of speech, too slow to talk with.

### settings.toml

serve.py reads `settings.toml` next to it:

```
voice = "hikari_voice.wav"
transcript = "It was a really quiet day here. ..."
model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
```

`voice` is the reference clip, a path relative to `settings.toml` or absolute. `transcript` is exactly what the clip says. `temperature` is optional and defaults to 0.7, see calibrating your voice below. `model` is optional and takes any qwen3-tts base checkpoint. without it, serve.py uses `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` on apple silicon and `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on a gpu. i tested `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` and `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` on the mac and `Qwen/Qwen3-TTS-12Hz-1.7B-Base` on the 4090, numbers are in the table above.

then ask it to talk:

```
curl -s localhost:8001/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input": "hi, this is my voice now."}' > out.pcm
```

it returns raw 16-bit mono pcm at 24 khz. `seed` is optional and makes a line come out the same every time, on the mac and on the 4090. to play it:

```
ffplay -f s16le -ar 24000 -ac 1 out.pcm
```

## using your own voice

point `voice` in `settings.toml` at about 15 seconds of clean speech (wav, mp3 or flac, mono or stereo, any sample rate) and put its exact transcript in `transcript`. qwen copies whatever the clip does, so pick takes that sound the way you want every sentence to sound. if every line in the clip ends going up, every line it says will too.

serve.py also turns "nya" into "N'yah", because hikari says it a lot and qwen kept saying "yeah" instead. take those two lines out if you don't need them.

### calibrating your voice

after you set your own voice, run this once:

```
.venv/bin/python calibrate.py
```

on windows it's `.venv\Scripts\python calibrate.py`.

it reads 10 test lines 30 times each at temperatures 0.9, 0.8, 0.7 and 0.6, and counts growls in each take. a growl is any tenth of a second where the pitch drops below 0.6x your clip's median pitch, so it works for low voices too. it prints the growl count and pitch range for each temperature, then writes the highest temperature with no growls (or the fewest, if none hit zero) as `temperature = ...` in `settings.toml`. the rest of the file stays the same. restart serve.py to use it.

it took 26 minutes on the 4090 under windows and 32 minutes on the mac mini (while my other server was using the same gpu). hikari's default of 0.7 came out of the same test: at 0.9, 0.8, 0.7 and 0.6 her voice growled 1, 1, 0 and 4 times out of 300 on the mac, and 4, 3, 1 and 4 times on the 4090.

`compare.py <model> ...` reads one line with each qwen3-tts model you give it and prints the timing, if you want to try other sizes. it uses mlx-audio, so it's mac only.

## hikari's voice

i designed her voice with elevenlabs voice design (from a written description, not anyone's recording), recorded a clip of it, and had qwen3-tts clone that. the clip in this repo is qwen's output. it's fine to use for your own projects.

## credits

- [qwen3-tts](https://github.com/QwenLM/Qwen3-TTS) (apache-2.0)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) (mit)
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (mit)
- [librosa](https://github.com/librosa/librosa) (isc), for calibrate.py

the qwen3-tts model weights are apache-2.0 too. this repo doesn't include them, serve.py downloads them from hugging face on first start.

mit license, see `LICENSE`.
