# hikari voice

a small speech server that clones a voice from a 15 second clip with qwen3-tts. it runs on apple silicon macs, and on nvidia or amd gpus. intel macs aren't supported. i run it as the voice of hikari, my stackchan robot. it speaks every qwen3-tts language (english, japanese, chinese, korean, german, french, russian, portuguese, spanish, italian) and streams audio as it goes.

i wrote it because the stock way of running qwen3-tts through mlx-audio used more memory and started speaking slower than i wanted on a mac mini that also runs everything else.

to set it up, skip to [running it](#running-it).

## what it does differently

on the mac, all of this keeps the audio byte-identical to plain mlx-audio with the same text, seed and temperature. i checked english, japanese and german lines before and after, on all three mac models below.

- the text embedding table is 622 mb, and a sentence only touches a few dozen of its rows. serve.py writes the table to its own `text_table-<model>.u16` file once and reads rows from that file on demand instead of keeping it in ram.
- mlx-audio recomputes the speaker embedding from the reference clip for every sentence. serve.py computes it once and reuses it.
- after the first sentence, the reference audio encoders aren't needed anymore (their output is cached), so serve.py drops them.
- qwen3-tts sometimes cuts off the last word. adding ` ^.` to the end of the text fixes most of it, and the model doesn't read it out. on my test set it took cut-off endings from 7 out of 24 lines to 1.
- at the default temperature of 0.9, every now and then a sentence ends an octave low, like a growl. serve.py samples at 0.7 instead. over 200 takes each, that took growls from 4 to 0 on a 4090 and from 2 to 0 on the mac, with the same speed. a longer 300-take run on the 4090 still caught one at 0.7, so they're rarer, not gone. her pitch moves a little less (about 0.4 semitones of range).

on a gpu, serve.py uses [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (cuda graphs), which turns the reference clip into a voice prompt once and reuses it, and streams in smaller chunks so the first audio comes out sooner. like on the mac, serve.py keeps the text table in that file instead of in vram, drops the reference audio encoders after the first sentence, and then hands the gpu memory pytorch reserved during warm-up back. on the 4090 those three keep the audio byte-identical to serve.py without them. the audio there isn't byte-identical to plain faster-qwen3-tts, it's just the same voice.

same 4 lines and seeds for each row, plain library vs serve.py:

| machine | model | | first audio | speed (rtf) | ram | vram | total | peak total |
|---|---|---|---|---|---|---|---|---|
| mac mini | 0.6b 8-bit | plain | 281 ms | 0.37 | 2223 mb | shared | 2223 mb | 4598 mb |
|  |  | this | 271 ms (-4%) | 0.37 (0%) | 1255 mb | shared | 1255 mb (-44%) | 1594 mb (-65%) |
| mac mini | 0.6b 4-bit | plain | 285 ms | 0.31 | 1956 mb | shared | 1956 mb | 4325 mb |
|  |  | this | 240 ms (-16%) | 0.29 (-4%) | 988 mb | shared | 988 mb (-49%) | 2034 mb (-53%) |
| mac mini | 1.7b 8-bit | plain | 371 ms | 0.49 | 3287 mb | shared | 3287 mb | 5701 mb |
|  |  | this | 373 ms (+1%) | 0.50 (+1%) | 2316 mb | shared | 2316 mb (-30%) | 3070 mb (-46%) |
| rtx 4090, linux | 0.6b | plain | 255 ms | 0.25 | 3040 mb | 3748 mb | 6788 mb | 7208 mb |
|  |  | this | 188 ms (-26%) | 0.26 (+5%) | 3043 mb | 3920 mb | 6963 mb (+3%) | 7438 mb (+3%) |
| rtx 4090, linux | 1.7b | plain | 279 ms | 0.28 | 3039 mb | 5836 mb | 8875 mb | 9218 mb |
|  |  | this | 206 ms (-26%) | 0.29 (+4%) | 3045 mb | 5792 mb | 8837 mb (0%) | 9309 mb (+1%) |
| rtx 4090, windows | 0.6b | plain | 380 ms | 0.33 | 2264 mb | 3791 mb | 6055 mb | 6406 mb |
|  |  | this | 297 ms (-22%) | 0.34 (+4%) | 2257 mb | 2719 mb | 4976 mb (-18%) | 5153 mb (-20%) |
| rtx 4090, windows | 1.7b | plain | 430 ms | 0.39 | 2256 mb | 5626 mb | 7882 mb | 8392 mb |
|  |  | this | 325 ms (-24%) | 0.40 (+3%) | 2251 mb | 4703 mb | 6954 mb (-12%) | 7123 mb (-15%) |

the percentages compare this server to plain. lower is better in every column. rtf is seconds spent per second of audio, so lower is faster. memory is counted the same way on every row: everything the process holds, in ram and on the gpu. a mac shares one pool of memory between the cpu and gpu, so its ram number already includes the model. on the 4090, total is ram plus vram. peak total adds the ram peak and the vram peak, which may not land at the same moment, so it can read a little high. the mac 1.7b and windows rows average 3 runs each, taking turns with plain, with two exceptions on windows 1.7b: plain uses the first 2 runs because a game started on the gpu in run 3, and this server's memory uses the last 2 because the gpu memory reading before run 1 was off (its timing uses all 3). the other rows are one run each. the smaller chunks cost 3-5% speed on the 4090, about the same as the run-to-run noise, and buy 70-105 ms sooner first audio. the linux rows are from before the gpu path got the mac's memory tricks. on windows those tricks take total memory 12-20% below plain. the ~2.3 gb of normal ram on the 4090 is cuda and pytorch libraries, not the model. windows doesn't report vram per process, so its vram numbers are total gpu use minus what the desktop used before the start, give or take 300 mb. the same card runs about 25-30% slower on windows than on linux either way.


### how i got there

i started by measuring where the memory actually went instead of looking at the model size. the text table was the biggest thing sitting in ram that a sentence barely touches: it has a row for each of qwen's 151,936 tokens across every language, and a sentence uses a few dozen. moving it to a file took about 600 mb out of ram without changing a single sample, since the numbers are the same. the speaker embedding and the encoders were the same story on a smaller scale, work the server kept repeating or kept around after it stopped needing it. dropping the encoders alone frees about 330 mb.

a few things i tried didn't make it:

- 4-bit weights are smaller and a bit faster, but the audio changes, and i wanted hikari to sound exactly the same.
- compressing the weights with xz or zstd only shrank them 12-18% on disk and saved no ram at all.
- splitting the table into per-language packs didn't help much. english alone needs about half the tokens, so the table only went from 622 to 315 mb.
- pocket-tts, a smaller model, was 4x faster, but her pitch fell to 93 hz on one line (she normally sits around 280) and it can't speak japanese.

## running it

get the code first:

```
git clone https://github.com/levinkin/hikari-voice
cd hikari-voice
```

then follow the steps for your machine below. it comes set up with hikari's voice, so it talks right away. switch to your own voice later in `settings.toml`.

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

### once it's running

serve.py listens on `127.0.0.1:8001`, so only your own machine can reach it. it's ready when it prints `Uvicorn running on http://127.0.0.1:8001`. ctrl+c stops it.

there's no cpu mode. without an apple silicon mac or a gpu, serve.py stops with `hikari-voice needs an apple silicon mac or an nvidia/amd gpu`. plain qwen-tts does run on cpu, but on a 32-thread desktop it took 13 s before any audio and 4.3 s per second of speech, too slow to talk with.

### settings.toml

serve.py reads `settings.toml` next to it:

```
voice = "hikari_voice.wav"
transcript = "It was a really quiet day here. ..."
```

you can add two optional lines:

```
temperature = 0.7
model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
```

`voice` is the reference clip, a path relative to `settings.toml` or absolute. `transcript` is exactly what the clip says. `temperature` is optional and defaults to 0.7, see calibrating your voice below. `model` is optional and takes any qwen3-tts base checkpoint. without it, serve.py uses `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` on apple silicon and `Qwen/Qwen3-TTS-12Hz-0.6B-Base` on a gpu. i tested `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` and `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` on the mac and `Qwen/Qwen3-TTS-12Hz-1.7B-Base` on the 4090, numbers are in the table above.

then ask it to talk, from a second terminal:

```
curl -s localhost:8001/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input": "hi, this is my voice now."}' > out.pcm
```

in windows powershell:

```
Invoke-WebRequest http://localhost:8001/v1/audio/speech -Method Post -ContentType application/json -Body '{"input": "hi, this is my voice now."}' -OutFile out.pcm
```

it returns raw 16-bit mono pcm at 24 khz. `seed` is optional and makes a line come out the same every time, on the mac and on the 4090. to play it (ffplay comes with ffmpeg):

```
ffplay -f s16le -ar 24000 -ac 1 out.pcm
```

restart serve.py after you change `settings.toml`.

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
