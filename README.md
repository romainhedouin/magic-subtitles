# magic-subtitles

A Claude skill for generating subtitles that match a film's **dubbed audio**,
with frame-accurate timing.

## The problem

Subtitle files for dubbed films are usually translated from the *original*
script, so they don't match what the voices actually say. Running Whisper
over the audio fixes the wording but introduces new problems: timings drift,
proper nouns come out mangled, and a single Whisper pass has no way to tell
you when it's confidently wrong.

## The approach

1. **Transcribe the audio three times, with three unrelated model families** —
   Whisper `large-v3` (Metal), NVIDIA's Canary-1B-v2, and Qwen3-ASR-1.7B. Where
   all three agree, the line needs no more thought; where they disagree,
   that's the audio telling you it's genuinely ambiguous. This is what lets
   you stop scrutinizing 90%+ of the file.
2. **Forced-align to word level** with wav2vec2, recovering per-word timing
   that Whisper's own segment timestamps don't reliably have.
3. **Rebuild the cues** from those word timings under real subtitle
   constraints, rather than trusting Whisper's own segment boundaries.
4. **Correct the transcript against a subtitle found for the film** — a real
   human translation. It's a *semantic parallel text*, not ground truth to
   copy from: in a reference run only ~54% of words overlapped the dub's
   actual wording. It's used to work out what a garbled word must have been
   and to fix proper-noun spelling, never to overwrite the dub's phrasing.
5. **Hand what's left to the user.** Some lines survive three independent
   ASR passes and a human-translation cross-check and are still wrong — not
   every error is recoverable from text alone. The skill reports what it
   couldn't verify, with its best guess and the evidence, rather than
   guessing silently.

See [SKILL.md](SKILL.md) for the full process and `scripts/` for the tooling.

## Usage

Invoke the skill and answer three questions: the dub's language, whether you
already have a reference subtitle (it searches either way), and whether to
just deliver the `.srt`, soft-mux it, or hard-burn it into the picture.

## Requirements

```bash
brew install ffmpeg
uv tool install mlx-whisper      # transcription, Metal
uv tool install mlx-audio        # Canary + Qwen3-ASR, Metal
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install whisperx  # word alignment
```

Apple Silicon only, for the Metal-accelerated passes.

## Development

The scripts are linted with [ruff](https://docs.astral.sh/ruff/); see
`ruff.toml` for the (deliberately narrow) rule set.

```bash
uvx ruff check scripts/
```

## License

MIT — see [LICENSE](LICENSE).
