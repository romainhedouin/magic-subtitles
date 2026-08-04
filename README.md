# magic-subtitles

A Claude skill for generating subtitles that match a film's **dubbed audio**,
with frame-accurate timing.

## The problem

Subtitle files for dubbed films are usually translated from the *original*
script, so they don't match what the voices actually say. Running Whisper over
the audio fixes the wording but introduces new problems: timings drift by
seconds, proper nouns come out mangled and inconsistent, and long stretches of
speech collapse into unreadable 20-second blocks.

## The approach

1. **Transcribe with WhisperX**, which adds forced alignment — a wav2vec2 model
   pins every *word* to the audio, instead of inferring timestamps from decoder
   attention. This is what fixes timing drift.
2. **Rebuild the cues** from those word timings under real subtitle constraints,
   rather than using whisper's own segment boundaries.
3. **Correct the transcript against a reference subtitle track extracted from
   the movie file itself.** Almost every release ships subtitles in the dub
   language: a real human translation, timecoded to this exact file. Nothing
   else fixes proper nouns nearly as reliably.

The reference is a *semantic parallel text*, not ground truth to copy from. The
two translations typically differ substantially in wording — in the reference
run, only ~54% of words overlapped. It's used to work out what a garbled word
must have been, never to overwrite the dub's phrasing.

## Usage

Invoke the skill and answer three questions: which Whisper model (quality vs.
time), which language, and whether to embed the subtitles into the video.

See [SKILL.md](SKILL.md) for the full process, and `scripts/` for the tooling.

## Requirements

```bash
brew install ffmpeg tesseract tesseract-lang hunspell
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install whisperx
```

## Results from the reference run

A 88-minute animated film, French dub, on an M4 Mac:

| | |
|---|---|
| Output | 769 cues, 0 overlaps, median 2.3s / 16.9 CPS |
| Timing | word-level forced alignment, sub-second boundaries |
| Coverage | 96% of reference dialogue moments |
| Corrections | 54 cues fixed against the reference, 0 timestamps altered |

Weakest area: musical numbers, where sung vocals defeat voice-activity
detection and the two translations diverge most, so fewer errors can be
confirmed. The skill reports what it could not verify rather than guessing.
