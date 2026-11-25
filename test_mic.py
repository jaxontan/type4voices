"""
Quick Microphone and Speech Recognition Diagnostic Test.
Records 3 seconds from the default microphone, saves to test_mic.wav,
and transcribes with Qwen3-ASR-1.7B.
"""
import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from qwen_asr import Qwen3ASRModel

SAMPLE_RATE = 16000
DURATION = 3.0
OUTPUT_WAV = "test_mic.wav"

print("=" * 60)
print("Microphone Diagnostic Test")
print("=" * 60)

devices = sd.query_devices()
default_in = sd.default.device[0]
print(f"Default input device: #{default_in} - {devices[default_in]['name']}")

print(f"\nRecording {DURATION:.0f} seconds in:")
for i in [3, 2, 1]:
    print(f"  {i}...")
    time.sleep(1)

print("\n>>> RECORDING! Speak a sentence now (e.g. 'Hello, this is a test')...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()
print("<<< Recording finished!")

peak = float(np.max(np.abs(audio)))
rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"\nAudio statistics:")
print(f"  Peak amplitude: {peak:.4f} (expected > 0.05 for clear speech)")
print(f"  RMS volume:     {rms:.4f}")

if peak < 0.01:
    print("\n[WARNING] Audio level is very low or silent! Check microphone mute switch or Windows Privacy Settings.")
else:
    print("  Microphone signal looks healthy!")

sf.write(OUTPUT_WAV, audio, SAMPLE_RATE)
print(f"Saved audio to: {OUTPUT_WAV}")

import torch

device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
print(f"\nLoading Qwen3-ASR model on {device} for verification...")
t0 = time.time()
load_kwargs = {"device_map": device, "dtype": torch.bfloat16} if device != "cpu" else {}
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", **load_kwargs)
print(f"Model ready ({time.time() - t0:.2f}s). Transcribing...")

t0 = time.time()
results = model.transcribe(audio=OUTPUT_WAV)
elapsed = time.time() - t0

if results:
    text = results[0].text.strip()
    lang = getattr(results[0], "language", "auto")
    print(f"\nResult [{lang}]: \"{text}\" (transcription time: {elapsed:.2f}s)")
else:
    print("\nNo result returned.")
