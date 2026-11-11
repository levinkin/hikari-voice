# hikari voice

a small speech server for apple silicon that clones a voice from a 15 second clip with qwen3-tts. i run it as the voice of hikari, my stackchan robot. it speaks every qwen3-tts language (english, japanese, chinese, korean, german, french, russian, portuguese, spanish, italian) and streams audio as it goes.

i wrote it because the stock way of running qwen3-tts through mlx-audio used more memory and started speaking slower than i wanted on a mac mini that also runs everything else.

## what it does differently

all of this keeps the audio byte-identical to plain mlx-audio with the same text and seed. i checked english, japanese and german lines before and after.

- the text embedding table is 622 mb, and a sentence only touches a few dozen of its rows. serve.py writes the table to `text_table.u16` once and reads rows from that file on demand instead of keeping it in ram.
- mlx-audio recomputes the speaker embedding from the reference clip for every sentence. serve.py computes it once and reuses it.
- after the first sentence, the reference audio encoders aren't needed anymore (their output is cached), so serve.py drops them.
- qwen3-tts sometimes cuts off the last word. adding ` ^.` to the end of the text fixes most of it, and the model doesn't read it out. on my test set it took cut-off endings from 7 out of 24 lines to 1.

on my mac mini, with the 8-bit 0.6b model:

| | plain | this |
|---|---|---|
| process memory | 2183 mb | 1292 mb |
| first audio | ~0.36 s | ~0.30 s |
| speed | unchanged | unchanged |

disk use doesn't change. the table just moves into its own file.

## running it

needs an apple silicon mac and python 3.12.

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python serve.py
```

the first start downloads `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` and writes `text_table.u16` (about 600 mb). that first run holds the table in memory too, so restart it once to get the lower number.

then ask it to talk:

```
curl -s localhost:8001/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input": "hi, this is my voice now."}' > out.pcm
```

it returns raw 16-bit mono pcm at 24 khz. `seed` is optional and makes a line come out the same every time. to play it:

```
ffplay -f s16le -ar 24000 -ac 1 out.pcm
```

## using your own voice

replace `hikari_voice.wav` with about 15 seconds of clean speech, and put its exact transcript in `hikari_voice.txt`. qwen copies whatever the clip does, so pick takes that sound the way you want every sentence to sound. if every line in the clip ends going up, every line it says will too. delete `text_table.u16` only if you change the model.

serve.py also turns "nya" into "N'yah", because hikari says it a lot and qwen kept saying "yeah" instead. take those two lines out if you don't need them.

`compare.py <model> ...` reads one line with each qwen3-tts model you give it and prints the timing, if you want to try other sizes.

## hikari's voice

i designed her voice with elevenlabs voice design (from a written description, not anyone's recording), recorded a clip of it, and had qwen3-tts clone that. the clip in this repo is qwen's output. it's fine to use for your own projects.

## credits

- [qwen3-tts](https://github.com/QwenLM/Qwen3-TTS) (apache-2.0)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) (mit)

mit license, see `LICENSE`.
