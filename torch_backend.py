import torch
from faster_qwen3_tts import FasterQwen3TTS


def load(model_name, voice_path, reference_text):
    voice = (FasterQwen3TTS.from_pretrained(model_name), str(voice_path), reference_text)
    list(speak(voice, "Hello there.", None))
    return voice


def speak(voice, text, seed):
    model, voice_path, reference_text = voice
    if seed is not None:
        torch.manual_seed(seed)
    for audio, _, _ in model.generate_voice_clone_streaming(text, "Auto", ref_audio=voice_path, ref_text=reference_text, append_silence=False, chunk_size=8, temperature=0.7):
        yield audio
