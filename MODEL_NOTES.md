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

## Measured performance (Core Ultra 5 228V, 2026-11-15)

Same clips, same prompt, after warm-up:

| Audio | CPU (torch +cpu, fp32) | Arc 130V GPU (torch +xpu, bf16) |
|---|---|---|
| 3.0 s | 4.6–5.2 s | 0.6–1.0 s |
| 4.2 s | 6.7–7.4 s | 1.05–1.1 s |
| 30 s | ~20 s | — |

The first GPU inference after loading compiles kernels (~5 s); the app's
warm-up pays that at startup with the real prompt and a dictation-sized clip.

`"torch_device": "auto"` (default in `config.json`) uses the GPU when torch can
see one and falls back to the CPU if loading there fails. To enable the Arc GPU:

```
winget install Microsoft.VCRedist.2015+.x64      # the xpu wheel needs it
.venv\Scripts\python.exe -m pip install "torch==2.15.0+xpu" --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
.venv\Scripts\python.exe dictate.py --doctor --full   # expect "compute xpu"
```

Without the VC++ redistributable, `import torch` fails with
`WinError 126 ... shm.dll`: the CPU wheel bundles `vcruntime140_1.dll`, the
xpu wheel does not.

---

## English repair: pipeline and model choice (2026-11-16)

Per dictation: loudness boost -> ASR twice in one batched call (with and
without the accent prompt, +0.15 s) -> `corrections.txt` -> LLM rewrite with
both hearings and the last two dictations (3 min) as topic context ->
guardrails -> paste.

`eval_rewrite.py` scores models on 28 weak-English cases (grammar, misheard
words, second hearing, context, traps, numbers), checking the text that would
actually be pasted:

The full comparison of every tested model (scores, timings, failure notes) is in
README.md → *Model comparison*. Short version: two fixers, swappable in the
tray. `Qwen/Qwen3-1.7B` (26/28, 1.0 s, in-process) is the default;
`google/gemma-4-E2B-it` (28/28, 1.44 s) runs in `.venv-t5` through
`rewrite_worker.py` because it needs transformers 5. Its 4.7 GB per-layer
embedding table stays in RAM and its audio/vision encoders are dropped, so it
takes 4.6 GB of GPU memory instead of 10.2 GB (the full model next to the
speech model hit `UR_RESULT_ERROR_DEVICE_LOST` on the Arc's shared 16 GB).

Guardrails reject a rewrite (raw transcript is pasted instead) when it: is
empty; grows past 1.6x / shrinks below 0.5x the words; loses a number; adds
paragraphs; switches to CJK script; keeps under 60% of the content words
(answers, jokes, copied examples); or copies the earlier dictation.

Add your own failures from `dictate.log` as cases in `eval_rewrite.py` and
re-run before changing models or the prompt.

The speech model writes what it hears, including misheard words and the
speaker's grammar. `rewrite.py` passes each transcript (3+ words) through a
local LLM on the same GPU that rewrites it into the clean English the speaker
meant. Toggle it in the tray ("Fix my English") or with `"rewrite_enabled"`.

- Download once: `.venv\Scripts\python.exe download_model.py --model Qwen/Qwen3-4B-Instruct-2507` (~8 GB)
- Arc 130V, bf16: ~2.2-2.6 s per sentence. `"rewrite_model": "Qwen/Qwen3-1.7B"`
  is ~1.2 s but worse at sound-alikes ("data bass" -> "data connection").
- Guardrails (`accept_rewrite`): a rewrite that loses a number, grows past
  1.6x or shrinks below 0.5x the word count, or adds paragraphs (an answer
  instead of a rewrite) is discarded and the raw transcript is pasted.
- The log keeps both: `transcribed ...` is what you said, `fixed English ...`
  is what was pasted.

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
which is why `min_seconds` defaults to 0.5 and a clip with no loudness
movement is dropped before transcription. Pin the language if you only ever
dictate in one.
