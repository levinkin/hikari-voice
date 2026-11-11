# hikari voice

a small speech server that clones a voice from a 15 second clip with qwen3-tts. it runs on apple silicon macs, and on nvidia or amd gpus. intel macs aren't supported. i run it as the voice of hikari, my stackchan robot. it speaks every qwen3-tts language (english, japanese, chinese, korean, german, french, russian, portuguese, spanish, italian) and streams audio as it goes.

i wrote it because the stock way of running qwen3-tts through mlx-audio used more memory and started speaking slower than i wanted on a mac mini that also runs everything else.

## what it does differently

on the mac, all of this keeps the audio byte-identical to plain mlx-audio with the same text and seed. i checked english, japanese and german lines before and after, on all three mac models below.

- the text embedding table is 622 mb, and a sentence only touches a few dozen of its rows. serve.py writes the table to its own `text_table-<model>.u16` file once and reads rows from that file on demand instead of keeping it in ram.
- mlx-audio recomputes the speaker embedding from the reference clip for every sentence. serve.py computes it once and reuses it.
- after the first sentence, the reference audio encoders aren't needed anymore (their output is cached), so serve.py drops them.
- qwen3-tts sometimes cuts off the last word. adding ` ^.` to the end of the text fixes most of it, and the model doesn't read it out. on my test set it took cut-off endings from 7 out of 24 lines to 1.

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

rtf is seconds spent per second of audio, so lower is faster. the mac 1.7b rows are the average of 3 runs each, taking turns with plain, while my other server was using the same gpu. the other rows are one run each. on the 4090 the process also uses about 3 gb of normal ram either way. the smaller chunks cost about 5% speed there, but it still talks 3x faster than real time and starts about 70 ms sooner. the extra vram is pytorch holding on to memory it reserved, not memory the model uses.


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
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python serve.py
```

on linux, pip pulls the cuda build of torch. for amd, install the rocm build of torch 2.15.0 from pytorch.org first, then run the pip line above. on windows, pip's default torch has no gpu support, so install the cuda build from pytorch.org first. i only tested linux with an nvidia rtx 4090: install took 81 s, startup about 7 s once the model is downloaded. the first start downloads `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.

there's no cpu mode. without an apple silicon mac or a gpu, serve.py stops with `hikari-voice needs an apple silicon mac or an nvidia/amd gpu`. plain qwen-tts does run on cpu, but on a 32-thread desktop it took 13 s before any audio and 4.3 s per second of speech, too slow to talk with.

### settings.toml

serve.py reads `settings.toml` next to it:

```
voice = "hikari_voice.wav"
transcript = "It was a really quiet day here. ..."
model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
```

`voice` is the reference clip, a path relative to `settings.toml` or absolute. `transcript` is exactly what the clip says. `model` is optional and takes any qwen3-tts base checkpoint. without it, serve.py uses `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` on apple silicon and `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on a gpu. i tested `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` and `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` on the mac and `Qwen/Qwen3-TTS-12Hz-1.7B-Base` on the 4090, numbers are in the table above.

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

`compare.py <model> ...` reads one line with each qwen3-tts model you give it and prints the timing, if you want to try other sizes. it uses mlx-audio, so it's mac only.

## hikari's voice

i designed her voice with elevenlabs voice design (from a written description, not anyone's recording), recorded a clip of it, and had qwen3-tts clone that. the clip in this repo is qwen's output. it's fine to use for your own projects.

## credits

- [qwen3-tts](https://github.com/QwenLM/Qwen3-TTS) (apache-2.0)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) (mit)
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (mit)

mit license, see `LICENSE`.
