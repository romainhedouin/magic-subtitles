# Why `large-v3` on Metal, unconditionally

This is background for the fixed decision in Step 0: **always run `large-v3`
on Metal via `mlx-whisper`, never ask, never offer `turbo`/`medium`/`small`.**
Read this file only if that decision needs justifying (to yourself or the
user) or you need the exact numbers — the main skill body does not inline it,
so a normal run of this skill never pays for reading it.

### Why there's no real choice to offer

**Run `large-v3` on Metal via `mlx-whisper`, not on CPU via CTranslate2.**
Same weights as CPU, same accuracy, measured **7× the speed** — Metal removed
the old speed-vs-accuracy trade entirely, so there is no real choice left to
hand the user here. Times below are for a ~90-minute film on an M4 Air.

| Model | Metal (`mlx-whisper`) | CPU (CTranslate2) | Notes |
|---|---|---|---|
| `large-v3` | **~12 min** | 1–2 h | Best accuracy. **Use this.** |
| `large-v3-turbo` | ~5 min | 20–40 min | Distilled decoder. Measurably worse on names and rare words. |
| `medium` | ~5 min | 20–40 min | Weaker again. |
| `small` | ~3 min | 10–20 min | Draft quality only. |

Because Metal makes `large-v3` cheap, the old speed-vs-accuracy trade is mostly
gone: there is now no good reason to accept a worse model to save ten minutes.

**Never use `turbo`.** Its distilled decoder gives up accuracy precisely on
proper nouns and rare vocabulary — the words that matter most and are hardest
to spot afterwards (measured on the reference run, same audio: `les eux`
where `large-v3` read `les Huns`). On Metal it saves about seven minutes, so
there is nothing to weigh.

### Measured speeds — pull these only if you need to give the user an estimate

**All figures measured on an M4 Air (16 GB), 16 kHz mono WAV, French audio.**
To estimate: `wall seconds ≈ audio seconds ÷ multiple`. A 2 h film is 7200 s of
audio, so ÷7 ≈ 17 min, ÷15 ≈ 8 min, ÷33 ≈ 3.5 min, ÷1 ≈ 2 h. The Parakeet
multiple held within 2% across two different films, so treat these as reliable.
Add a **one-time ~5 min model download** per model, and ~10 s for ffmpeg audio
extraction.

| Model | Runtime | × realtime | 2 h film | Own timestamps? |
|---|---|---|---|---|
| `large-v3` | **`mlx-whisper` (Metal)** | **7.3×** | **~16 min** | Yes, segment-level |
| `large-v3` | WhisperX (CPU) | ~0.8–1.5× | 1.5–3 h | Yes, segment-level |
| Canary-1B-v2 | `mlx-audio` (Metal) | **15×** | ~8 min | **No — all zeros** |
| Qwen3-ASR-1.7B | `mlx-audio` (Metal) | ~7× | ~17 min | **No — one segment per call** |
| Parakeet TDT 0.6B v3 | `parakeet-mlx` (Metal) | 33× | ~3.5 min | Yes, incl. word-level |

The Canary/Qwen3 rows above predate silence-aligned windowing (Step 5b/5c now
target ~8s windows instead of fixed 30s blocks — see there for why). Expect
roughly 4x the model calls and a real but unmeasured slowdown; the per-call
overhead is small next to a 30s decode, so this is unlikely to change which
model is the bottleneck, but don't quote the 15×/~7× figures above as current.

The 7.3× for `mlx-whisper` **depends on the anti-loop flags** in Step 5a.
Without them it measures 5.2× *and* produces garbage — see there.
`faster-whisper`/WhisperX only load Whisper-architecture weights; other
models need their own MLX runtime, but the **forced-alignment stage is
model-agnostic**, so any ASR can feed it.

Parakeet is documented here but **not used by the pipeline** — Canary beat it
decisively on the same audio and Canary's 8 min is already cheap. Reach for
Parakeet only if you need a throwaway draft in under 4 minutes.

Accuracy and failure modes, measured on the same French film (1 h 52 m):

| | Parakeet v3 | Canary-1B-v2 |
|---|---|---|
| Words transcribed | 7 049 | **9 924** (~40% more speech caught) |
| Sample line | `Dégage toi les voilà` | `Ça y est, les voilà` (correct) |
| Sample line | `100 euros que les mettre dans le nom` | `100 euros que je les mets dans le vent` (correct) |
| Characteristic failure | 20 s+ mega-cues (worst: **233 s**); silent flips to English mid-film | decoder repetition loops (`Escorte. Escorte. Escorte.`) |

**Canary is the accuracy pick, Parakeet the speed pick.** Canary is clearly
better French, but `mlx-audio` returns `start=0.0, end=0.0` for every segment
— it has no segmentation at all, so it is only usable *with* the alignment
stage, and the audio must be pre-chunked with ffmpeg (its `generate` has no
chunking and defaults to `max_tokens=200`, which truncates anything longer
than ~30 s of speech).

Two `mlx-audio` traps:

- **It silently drops `--language`.** The CLI filters kwargs against the
  named parameters of each model's `generate`, and Canary takes
  `source_lang`/`target_lang`. So `--language fr` vanishes, the `en`/`en`
  defaults apply, and Canary returns an English **translation** of French
  audio. Always pass `--gen-kwargs '{"source_lang":"fr","target_lang":"fr"}'`,
  or use the Python API. Source and target must always be equal — this
  pipeline never wants translation.
- **Cohere-transcribe-03-2026 does not currently work on MLX**, despite
  topping the Open ASR leaderboard (5.42% WER). Both `beshkenadze`
  conversions (fp16 and 8-bit) emit token soup that is identical across
  different audio and equally broken in English, `mlx-community/…-mlx-8bit`
  is an empty repo, and `cohere_asr`'s VAD path crashes
  (`mx.array.astype(np.float32)`). Do not spend time on it; the untried
  routes are the `openasr` Rust binary and the CoreML/ONNX conversions.
