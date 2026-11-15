# Model Notes & Licensing Information

## Primary Model: Qwen3-ASR-1.7B
- **Repository / Source**: [https://huggingface.co/Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- **Official GitHub**: [https://github.com/QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- **License**: Apache-2.0 (permissive open source; commercial and local use allowed)
- **Parameters**: 1.7 Billion
- **Disk footprint**: ~4.4 GB in the local cache
- **Capabilities**:
  - Multilingual automatic speech recognition (52 languages & dialects)
  - Automatic language identification
  - High accuracy on Chinese (dialects & Mandarin), English, mixed Chinglish, and technical jargon
  - Supports offline file transcription and streaming
- **Local Cache Location**:
  `%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen3-ASR-1.7B`

## Measured performance (this laptop, CPU-only torch)

| Audio | Time to text | Realtime factor |
|---|---|---|
| 1.0 s | 5.9 s (first call, includes warm-up) | — |
| 2.0 s | 3.0 s | ~1.5× |
| 4.0 s | 4.5 s | ~1.1× |
| Model load (torch already imported) | 5.4 s | — |
| First launch of a fresh process | +25–40 s | torch/transformers import |

Installing the XPU build of torch is the single biggest speed-up available on
this hardware; see the *Troubleshooting* section of `README.md`.

---

## Alternative Model: Qwen3-ASR-0.6B (High Throughput / Ultra-Low Latency)
- **Repository / Source**: [https://huggingface.co/Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- **License**: Apache-2.0
- **Parameters**: 0.6 Billion (~1.2 GB disk footprint)
- **When to use**: if transcription latency matters more than accuracy.
- **Important**: `dictate.py` runs with Hugging Face offline mode forced on, so
  this model must be cached *before* switching. Download it first, then change
  the model:
  ```
  .venv\Scripts\python.exe download_model.py --model Qwen/Qwen3-ASR-0.6B
  ```
  then set `"model_id": "Qwen/Qwen3-ASR-0.6B"` in `config.json`.

---

## Optional Comparison Model: FireRedASR2-AED
- **Repository**: [https://github.com/FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)
- **Focus**: Mandarin / English speech recognition
- **License**: check the upstream repository before commercial use
- **Status**: not integrated. It would need the same `transcribe(audio) ->
  text` adapter that `VoiceDictationApp.transcribe()` provides for Qwen3-ASR.

---

## Language pinning

`config.json` accepts `"language": "Auto"` or any language name the model
reports (see `Qwen3ASRModel.get_supported_languages()`). `Auto` is convenient
but the language detector can flip on clips under ~1 second or on noisy audio,
which is why `min_seconds` defaults to 1.0. Pin the language if you only ever
dictate in one.
