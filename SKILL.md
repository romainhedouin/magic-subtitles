---
name: magic-subtitles
description: Generate subtitles that match a film's dubbed audio, using WhisperX for transcription plus forced alignment, then correcting the transcript against a reference subtitle track extracted from the movie file itself. Use when a user wants accurate subtitles for a movie, wants subtitles matching what the voices actually say (rather than a translation of the original script), or wants to fix badly-timed or mistranslated subtitles.
---

# Magic Subtitles

Produce a subtitle file that matches **what the voices actually say**, with
frame-accurate timing.

The naive approach — run Whisper, write the SRT — fails in three predictable
ways: timings drift by seconds, proper nouns are mangled, and long runs of
audio collapse into unreadable blocks. This skill fixes all three.

The core idea: **transcribe with WhisperX, then correct the transcript against
a reference subtitle track already inside the movie file.** Almost every video
release ships subtitles in the dub language. That track is a real human
translation, timecoded to this exact file. It is the single most valuable asset
in the process, and it is what makes the difference between "mostly right" and
"reliable".

---

## Step 0 — Interview the user

Ask these three questions **before doing any work**. Use the AskUserQuestion
tool, all three in a single call.

### Q1. Which Whisper model?

Estimates are for a ~90-minute film on Apple Silicon **CPU** (CTranslate2 has no
Metal backend, so the GPU is not used). With CUDA, divide by roughly 10.

| Model | Time | Notes |
|---|---|---|
| `large-v3` | **1–2 h** | Best accuracy. **Use this.** |
| `large-v3-turbo` | 20–40 min | Distilled decoder. Measurably worse on names and rare words. |
| `medium` | 20–40 min | Weaker again. |
| `small` | 10–20 min | Draft quality only. |

**Do not reach for `turbo` to save time.** It is a distilled decoder, and the
accuracy it gives up lands precisely on the words that matter most — proper
nouns and rare vocabulary — which are also the hardest for a human to spot and
fix afterwards. Measured on the reference run, same audio, same settings:

| | large-v3-turbo | large-v3 |
|---|---|---|
| "the Huns crossed the border" | `les eux` | `les Huns` (correct) |

Every hour `turbo` saves is repaid several times over in manual review, because
each surviving name error recurs throughout the film. Offer the faster models
only if the user explicitly accepts a rougher result, and say plainly that the
review burden grows.

### Q2. Which language is the audio in?

Default **French**. Offer **English** as the other suggestion. Accept anything
else the user types — the pipeline is language-generic, but you need the right
code in three places: Whisper (`fr`), tesseract (`fra`), hunspell (`fr`).

### Q3. Burn the subtitles into the video?

Three outcomes, and the user should understand the trade:

- **No (default)** — just the `.srt` file. Players load it automatically when
  it sits beside the video with a matching filename.
- **Soft-mux** — remux the subtitle track into the container. Fast, picture
  untouched, toggleable in the player. Almost always what people actually want.
- **Hard burn (pixel-embed)** — subtitles painted into the picture. Requires a
  full video re-encode: slow (1–3× realtime), lossy, permanent, and the file
  cannot be turned off. Only do this if the target device can't handle subtitle
  tracks.

Steer toward soft-mux if the user asks for burn-in without a specific reason.

---

## Step 1 — Gather context about the film

**This step is vital. Do not skip it, and do it before transcribing.** Nearly
every correction in Step 8 depends on knowing what the film is about. Without
it you will catch only the errors a dictionary flags, which is a small minority
of them.

Identify the film (filename, year, release tag) and pull together:

- a **detailed plot synopsis** — scene by scene if available
- the **character list with correct name spellings**, and the dub-language
  spellings if they differ
- place names, factions, invented terminology

Use WebSearch/WebFetch for a synopsis and cast list. This is ordinary factual
lookup and is not the same as scraping a full transcript.

Why it matters: ASR errors are usually *plausible-sounding* words. Only context
tells you that `les uns` should be `les Huns`, that `défaire Léa` should be
`défaire les Huns`, or that a garbled name is a character. Without the synopsis
you will only catch errors a dictionary flags — which is a small fraction of
them.

Write the names and terms into a working glossary. Every one is a term Whisper
will get wrong repeatedly and consistently.

## Step 2 — Probe the file

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name,channels:stream_tags=language,title \
  -of default=noprint_wrappers=1 "$MOVIE"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$MOVIE"
```

Record for later:
- the audio stream index in the target language, and its **channel count**
- any **subtitle** stream in the target language, and whether it is text
  (`subrip`, `ass`, `mov_text`) or bitmap (`hdmv_pgs_subtitle`, `dvd_subtitle`)
- the duration

**Check free disk space now** (`df -h .`). The pipeline writes a WAV of roughly
`duration × 32 kB/s` (~170 MB for 90 min) plus chunks of the same size again.
Running out mid-run corrupts outputs in confusing ways.

---

## Step 3 — Extract the audio

For a **5.1** track, take the **centre channel only** — dialogue is mixed there,
while score and effects sit in L/R/surrounds. This is free separation and it
measurably helps.

```bash
# 5.1 (6 channels): centre channel only
ffmpeg -y -v error -i "$MOVIE" -map 0:$AIDX -vn \
  -af "pan=mono|c0=c2" -ar 16000 -c:a pcm_s16le audio.wav

# stereo or mono: plain downmix
ffmpeg -y -v error -i "$MOVIE" -map 0:$AIDX -vn \
  -ac 1 -ar 16000 -c:a pcm_s16le audio.wav
```

Sanity-check the level. Speech should sit around −25 to −15 dB mean:

```bash
ffmpeg -hide_banner -i audio.wav -af volumedetect -f null - 2>&1 | grep mean_volume
```

**Do not** apply `dynaudnorm`, `highpass`/`lowpass` or other "cleanup" filters.
Tested: they pump and smear, and made recognition *worse*.

**Do not** run Demucs or other source separation by default. It targets music
masking, which only affects musical numbers, and its artifacts can degrade the
much larger body of clean dialogue. Consider it only if the user specifically
cares about song lyrics, and then only on those passages.

---

## Step 4 — Get the reference transcript

This is the step that makes the result reliable. **Do not skip it.**

The reference must be a genuine transcript in the **dub language**, not a
translation of the English script. A subtitle track from the release itself
qualifies; a machine translation of the original screenplay does not.

### 4a. Preferred: a subtitle track inside the movie file

Text-based — extract directly:

```bash
ffmpeg -y -v error -i "$MOVIE" -map 0:$SIDX reference.srt
```

Bitmap (PGS/VobSub) — extract and OCR:

```bash
ffmpeg -y -v error -i "$MOVIE" -map 0:$SIDX -c copy reference.sup
python3 scripts/pgs_to_srt.py reference.sup work/pgs fra   # tesseract lang code
python3 scripts/fix_ocr.py work/pgs/raw.srt reference.srt fr
```

OCR is good but not perfect. `fix_ocr.py` handles the systematic errors, and
`scripts/verify_ocr.py` lists remaining suspects. **Verify suspicious words
against the source bitmap before "fixing" them** — see Pitfalls.

### 4b. Optional: a supplementary reference found online

**Optional, but very valuable when you find one — spend real effort looking.**
If you can find a transcript or subtitle file **in the dub language** for this
film, use it alongside the in-file track. A second reference resolves cases
where the first is too divergent to judge, which is the single biggest cause of
errors that survive to the end.

Best target: a **SDH / "sourds et malentendants" / hearing-impaired** subtitle
track. Unlike an ordinary subtitle translation, SDH transcribes the *dub audio*
itself, so it matches what the voices say almost word for word. That is the one
artefact that would resolve nearly everything a normal reference cannot. Search
for it explicitly.

Ordinary subtitle files in the dub language are much weaker: they are another
independent translation, so they diverge from the dub in the same places the
first reference does, and add less than you would expect.

Two rules:

- It must be **dub-language**. A transcript of the original-language script,
  or a machine translation of one, is worse than useless: it will confidently
  disagree with the dub everywhere and generate false corrections.
- Use it to **verify and adjudicate**, never to bulk-replace the transcript.
  Do not copy a full screenplay into the output file; quote only what you need
  to settle a specific word.

**If you cannot find one, that is fine — carry on.** The in-file track plus the
Step 1 synopsis is enough. Just expect a slightly longer manual-review table at
the end, and say so.

### 4c. Fallback: ask the user

If the file has no subtitle track in the target language and you found nothing
online, ask the user to supply a reference subtitle file for this release. A
`.srt` they already have works fine.

If there is no reference at all, you can still transcribe — but say clearly that
verification was limited to the synopsis, grammar and internal consistency, and
expect materially more errors to survive.

### An important caveat to tell the user

The subtitle track and the dub are usually **different translations**. In this
skill's reference run, only ~54% of words overlapped, and the median per-cue
match was 20%. Same meaning, different wording.

So the reference is **not** ground truth to copy from. It is a *semantic
parallel text*: use it to work out what a garbled word must have been, never to
overwrite Whisper's wording wholesale. The dub transcript is what matches the
voices; that is the whole point.

---

## Step 5 — Transcribe with WhisperX, in chunks

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install whisperx
```

**Chunk the audio. Do not run WhisperX on a full-length film.** It loads the
entire waveform as float32 (~5.6 GB for 90 min) alongside the model, VAD and
alignment nets, and gets OOM-killed on a 16 GB machine. The kill surfaces as
exit code **137**, and a shell wrapper can mask it as "exit 0" — check for the
output file, not the exit status.

Split on silence so no cut lands mid-word:

```bash
python3 scripts/chunk_audio.py audio.wav work/chunks 600   # ~10-min targets
```

Then transcribe each chunk, **checkpointing per chunk**:

```bash
bash scripts/run_chunks.sh work/chunks work/srt large-v3 fr
```

`run_chunks.sh` skips any chunk whose SRT already exists. This matters: in the
reference run a process kill destroyed the wrapper but ~70 minutes of completed
work survived and the run resumed cleanly. Re-run the same command to resume.

Run it in the background and monitor completions rather than blocking.

---

## Step 6 — Recover word-level timings

WhisperX aligns to word level internally but **throws that away** when writing
SRT, leaving whisper's coarse segment boundaries. Recover it with an
alignment-only pass — a wav2vec2 forward pass, no re-transcription:

```bash
python3 scripts/align_words.py work/chunks work/srt work/words.json fr
```

---

## Step 7 — Build the subtitle file

```bash
python3 scripts/build_srt.py work/words.json work/chunks/cuts.txt draft.srt fr
```

Rebuilds cues from acoustic word onsets under real subtitle constraints:
max 6 s, max 84 characters over 2 lines of 42, splits on silences > 0.8 s and
after sentence-final punctuation, extends fast cues into silence to keep
reading speed under 20 CPS, and guarantees no overlap.

Language-aware punctuation: French requires a space before `! ? ; :`, English
does not. `build_srt.py` takes the language code for this reason.

Line breaks prefer sentence boundaries. Regrouping words into cues can strand a
capitalised sentence start mid-line with no punctuation ("...qui es-tu Je suis
un ami"), which reads badly. `build_srt.py` weights the break point to fall
there. It cannot fix every case -- a cue that is already two full lines has
nowhere to put a third -- so check the output for leftovers:

```bash
grep -vE '^[0-9]+$|-->' final.srt | grep -nE "[a-zà-ÿ] [A-ZÀ-Ý][a-zà-ÿ]"
```

Filter out proper nouns and noun phrases ("la Dame Marieuse", "du général Li")
before treating a hit as a real sentence start.

---

## Step 8 — Correct the transcript, line by line

This is the step that separates a usable file from a rough one. **A dictionary
check alone is not enough** — it only catches errors that spell as non-words.
Most ASR errors are perfectly valid words in the wrong place: `les uns` for
`les Huns`, `Meuf` for `Bœuf`, `la passe` for `la face`. Every one of those
passes a spell-check cleanly.

**The governing principle: the reference is a reliable error *detector* and an
unreliable error *corrector*.** It tells you a line is wrong far more often than
it tells you what is right, because it is a different translation. When the
reference shows the line is wrong but not what it should be, that is not a
failure — that is the normal case, and it belongs in the user-review table.

Real example: the reference read *Petite maladroite*, the ASR had
`Petit en petit`. Enough to prove the line was broken; not enough to recover
`Petite empotée`, which only the audio gave.

So run these passes, in this order.

### Pass A — dictionary sweep (cheap, catches the obvious)

```bash
python3 scripts/find_suspects.py draft.srt fr --foreign en
python3 scripts/adjudicate.py draft.srt reference.srt fr WORD1 WORD2 ...
```

### Pass B — contextual line-by-line read (mandatory, catches the rest)

Read **every cue** against the reference at the same timecode. Do not skip this
and do not sample it. Work in batches of 100–150 cues so each batch fits
comfortably in context:

```bash
python3 scripts/review_pairs.py draft.srt reference.srt 1 150
python3 scripts/review_pairs.py draft.srt reference.srt 151 300
# ... continue to the end
```

For each cue ask:

1. **Does it make sense** in this scene, given the synopsis from Step 1?
2. **Does it match the reference's meaning**, even if the wording differs?
3. **Is a proper noun involved?** Check it against the Step 1 glossary.
4. **Is it grammatical French/English?** Agreement errors betray misheard words
   (`Mulan est parti` for a female character; `Fa Zhou sera humiliée` for a male
   one).
5. **Would a viewer stumble on it?** If it reads as nonsense, it is nonsense.
6. **Does it join up with the next cue?** Sentences routinely span a cue
   boundary, and an error next to the split looks fine in isolation. Read
   consecutive cues joined together, not only one at a time. A cue ending
   `...qui va changer` reads acceptably alone; followed by
   `aucun doute leur sauter aux yeux !` it is plainly `qui va sans`.

Typical catches that only this pass finds:

| Whisper heard | Actually | How you know |
|---|---|---|
| `les uns` | `les Huns` | Synopsis: the Huns are the antagonists |
| `Meuf, porc, poulet` | `Bœuf, porc, poulet` | A food list — `Meuf` is nonsense here |
| `défaire Léa` | `défaire les Huns` | No character named Léa exists |
| `je ne perdrai pas la passe` | `... la face` | Fixed idiom |
| `Salut, chienne peau` | `Salut, Chien-Po` | Glossary name, spelled phonetically |
| `Jésus-Bren` | `Je suis prêt` | Meaningless; reference gives the sense |

Beware the inverse: `quelques-uns` is correct and must **not** be swept up by a
blanket `uns → Huns` replacement. Always anchor replacements to enough
surrounding words to be unambiguous.

### Pass C — hallucination sweep

Whisper emits boilerplate from its training data, especially over silence and
end credits. Always check:

```bash
grep -inE "sous-titrage|subtitle|amara|merci d'avoir regardé|thanks for watching|abonnez-vous" draft.srt
grep -vE '^[0-9]+$|-->|^$' draft.srt | sort | uniq -c | sort -rn | head
```

Any subtitling-house credit, "thanks for watching", or line repeated many times
is a hallucination. Drop it via `drop_matching` in the fixes file.

### Applying

```bash
python3 scripts/apply_fixes.py draft.srt final.srt fixes.json
```

`apply_fixes.py` **rewrites only text lines and never touches timestamps.** It
asserts this itself and reports any timing line that changed, plus every cue it
dropped.

If a replacement lengthens a line past 42 characters, re-wrap it — that changes
line breaks only, never timings.

### Judgement rules

- **Only fix what you can justify** from the reference, the synopsis, or a fixed
  idiom. If the translations diverge too far to tell, leave the word alone. A
  wrong word the user can spot beats a confident-sounding invention they cannot
  audit.
- Musical numbers diverge most between dub and subtitle translations, so that is
  where you will resolve least. Expect it.

### On running several models for "multiple sources"

Tempting, and mostly measured *not* to work. Record before repeating it:

- **Two Whisper variants are not independent.** They share weights and training
  data, so they reproduce each other's errors. Confirmed directly: whisper.cpp
  `large-v3` independently produced `Chifou` — the same error the CTranslate2
  `large-v3` run made, which had already been corrected. A second Whisper pass
  will confidently agree with the first that `les Huns` is `les uns`.
- **A different architecture is genuinely independent but weak.** The wav2vec2
  CTC pass (Pass D) is architecturally unrelated, and still found 0 new errors.
- **Do not run them concurrently on CPU.** CTranslate2 already saturates the
  cores; three parallel runs are roughly three times slower each, with no
  wall-clock gain. Run sequentially, or in parallel only with a GPU.

What remains untested, and is the version worth trying if you want to revisit
this: a **three-way vote** across genuinely diverse *and* individually strong
systems — e.g. `large-v3`, a French-fine-tuned Whisper, and a CTC model — taking
majority agreement rather than pairwise disagreement. Two-way comparison against
a weak second system is what failed here; that is a narrower result than "model
diversity doesn't help".

Spend the effort on Step 1 (synopsis) and Pass B (line-by-line) first. Both
were measured to catch far more, for far less compute.

### Pass D — independent ASR cross-check (optional, measured low-yield)

The idea: transcribe the audio a second time with a *different model family*,
and treat disagreement as a hint. Unlike a reference translation, this compares
two readings of the same audio, so in principle it works inside songs and
ad-libs.

```bash
python3 scripts/asr_ctc.py work/chunks work/ctc.json \
  jonatasgrosman/wav2vec2-large-xlsr-53-french 6
python3 scripts/cross_check.py draft.srt work/ctc.json work/chunks/cuts.txt --min-len 6
```

**Measured result on the reference run — read this before spending time on it.**
Validated by running the detector against 14 errors already confirmed by other
means:

| | 30 s windows | 6 s windows |
|---|---|---|
| Known errors flagged | 7 / 14 | 7 / 14 |
| Cues flagged (of 777) | 255 (33%) | 234 (30%) |
| Known errors in top 50 | 0 | 1 |
| **New errors found** | — | **0** |

Half the known errors were missed, a third of all cues were flagged, and the
highest-ranked suspects were ordinary correct words (`hommes`, `cheval`,
`l'empereur`). The cause is structural: a CTC model has no language model, so
its output diverges from Whisper's nearly everywhere. When disagreement is the
base rate, disagreement carries little information.

Do not run this expecting results. It is kept because it is cheap (~7 min for a
90-minute film, since CTC is a single forward pass rather than autoregressive
decoding) and might do better with a stronger second model — a French-fine-tuned
Whisper, or a genuinely different family. **A second Whisper run is not a
substitute**: same weights reproduce the same errors, which was confirmed here
when whisper.cpp large-v3 independently produced `Chifou`, an error the first
pass had made too.

### Validate any detector before you trust it

The transferable lesson. Before believing a new error-detection technique,
**run it against errors you have already confirmed** and measure how many it
finds and how highly it ranks them. This turns "seems useful" into a number,
and it is the only thing that separates a real check from a plausible-looking
list. It cost one command here and prevented shipping a 234-cue list of noise
as though it were diligence.

### Hand the remainder to the user

Whatever you could not resolve, present as a table so the user can check it
against the audio in one pass. Always these four columns:

| Time | Whisper heard | Transcript says | Context |
|---|---|---|---|
| 00:28:05 | `un vrai purson` | *Torride, non ?* | Mushu boasting after a fire trick |
| 00:46:38 | `Kenny ! McKay !` | *Gengis Khanasson...* | Mushu naming the horse; likely a joke name |
| 00:56:13 | `Monkey !` | *(no reference cue)* | Shouted during the battle |

Sort by timestamp so the user can work through the film linearly. Give the
timestamp in `HH:MM:SS` so it can be typed straight into a player. In
**Context**, say what is happening in the scene — that is usually what lets the
user recognise the line without hunting for it.

Group by ease: short isolated shouts first, mid-sentence errors next, sung lines
last. Flag anything you believe is probably *correct* (real slang, a deliberate
dub invention) so the user does not waste time on it.

### The endgame is the user, and that is fine

Once the reference, the dictionary and the synopsis are exhausted, the remaining
errors are the ones where the dub says something no other text records. On the
reference run **every correction after that point came from the user listening**
— `pur-sang`, `rufian`, `blanc-bec`, `Petite empotée`, `porc aigre doux`. None
were recoverable from any reference.

Do not treat that as a shortfall to hide or grind against with more compute.
Deliver a short, well-ordered review table and say plainly that these need ears.
A precise 10-line table is worth more to the user than another hour of
processing that finds nothing.

---

## Step 9 — QA

```bash
python3 scripts/qa_srt.py final.srt reference.srt
```

Checks and reports:
- zero overlapping or non-monotonic cues, no zero-length cues
- no cue over 2 lines; reading speed distribution
- **coverage**: reference cues with no corresponding output cue, which finds
  dialogue the ASR missed entirely
- residual dictionary misses

Investigate any coverage gap of more than a few seconds. In the reference run
the single 18-second gap turned out to be a musical passage — legitimate, but
worth confirming rather than assuming.

---

## Step 10 — Deliver

Name the file to match the video so players auto-load it:

```
Movie.Name.2024.1080p.mkv
Movie.Name.2024.1080p.fr.srt
```

### Required: every video this skill writes carries AAC-LC stereo audio

Most releases ship **AC3 or E-AC3 5.1**. Android TV and Chromecast will not play
it — the picture arrives and the sound is silent, with nothing in the ffmpeg
output to warn you. Convert the audio. Never `-c:a copy` into a delivered video.

```bash
-c:a aac -profile:a aac_low -b:a 192k -ac 2
```

Keep the source sample rate (48 kHz for essentially every film). Resampling to
44.1 kHz buys no compatibility and only loses quality. If a device still refuses
the track, try the **MP4 container** next, not a different sample rate.

Soft-mux (picture untouched, subtitles toggleable):

```bash
ffmpeg -y -i "$MOVIE" -i final.srt -map 0:v:0 -map 0:a:0 -map 1 \
  -c:v copy -c:a aac -profile:a aac_low -b:a 192k -ac 2 \
  -c:s srt -metadata:s:s:0 language=fra "output.mkv"
```

Hard burn (only if the user explicitly chose it — slow, lossy, permanent):

```bash
ffmpeg -y -i "$MOVIE" -vf "subtitles=final.srt:force_style='FontSize=24'" \
  -map 0:v:0 -map 0:a:0 \
  -c:v libx264 -crf 18 -preset medium \
  -c:a aac -profile:a aac_low -b:a 192k -ac 2 "output-burned.mkv"
```

To fix the audio on a video that is otherwise already finished, **do not
re-encode the picture.** Stream-copy the video and convert only the audio —
under a minute for a feature film, and the video stays bit-identical:

```bash
ffmpeg -y -i "in.mkv" -map 0:v:0 -map 0:a:0 \
  -c:v copy -c:a aac -profile:a aac_low -b:a 192k -ac 2 "out.mkv"
```

Verify before handing it over:

```bash
ffprobe -v error -select_streams a:0 \
  -show_entries stream=codec_name,profile,channels,sample_rate -of csv=p=0 out.mkv
# expect: aac,LC,2,48000
```

Never overwrite the user's original. Write a new file.

---

## Pitfalls

Each of these cost real time in the reference run.

**Whisper hallucination loops.** Silence at the start of a film makes whisper
emit a filler line, which it then carries forward as decoder context for the
*entire* film. Symptom: hundreds of identical cues. Fix: disable carried
context (`-mc 0` in whisper.cpp, and WhisperX does not carry context by
default) and enable VAD. Always check for repeated lines:
`grep -vE '^[0-9]+$|-->|^$' out.srt | sort | uniq -c | sort -rn | head`

**OOM on long audio.** See Step 5. Chunk it.

**AC3 5.1 audio casts as silence.** The file plays perfectly on the desktop, so
it reads as a device fault rather than a file fault, and ffmpeg emits no warning.
Android TV and Chromecast simply drop the track: picture, no sound. Deliver
AAC-LC stereo (Step 10). When a user reports missing audio on a TV, probe the
codec *before* investigating anything else — and fix it by stream-copying the
video, not by re-encoding it.

**Masked exit codes.** A background wrapper can report success while the real
process was killed. Verify the output file exists and is non-empty.

**Don't trust a running process as evidence of progress.** Check timestamps in
the log, not just `pgrep`. A matching process may be a different stage, or a
stale one.

**OCR diacritics.** Tesseract misreads `û` as `ü`, drops circumflexes, and
renders `!` as `I`, `I!` or `!I`. `fix_ocr.py` handles these — but **verify
against the source bitmap before assuming**. In the reference run, `Ecoute`
looked like a dropped accent and was not: that track deliberately omits acute
accents on capitals while keeping the cedilla on `Ç`. Two independent OCR
passes with different `--psm`/`--oem` settings disagreed on only 3 of 927 cues,
so cross-passing is a cheap confidence check.

**Word lists that include ambiguous tokens.** When detecting foreign-language
cues, never include words that also exist in the target language. Adding `a`,
`on`, `or`, `in` to an English-detection list would delete the French line
"On a vu" — it scores 67% "English". Restrict such lists to unambiguous tokens
and always print what you dropped so it can be checked.

**Multi-word patterns must allow newlines.** Cue text contains line breaks, so
a replacement written with literal spaces silently fails to match:

```python
["Meuf, porc, poulet", "Boeuf, porc, poulet"]      # WRONG: misses "Meuf,\nporc, poulet"
["Meuf,\\s+porc,\\s+poulet", "Boeuf, porc, poulet"]  # right
```

This fails *silently* -- apply_fixes.py reports the cues it did change, not the
patterns that matched nothing. After applying, always grep the output for the
old text to confirm each fix actually landed.

**Spell-checker artifacts.** A regex like `[a-zà-ÿ]+` silently strips capitals,
turning `Habille` into `abille` and generating phantom errors. And hunspell
splits ligatures, so `bœuf` reports as `uf`. Check the extraction before
chasing the "error".

---

## Scripts

All under `scripts/`, all take explicit arguments, none hardcode paths.

| Script | Purpose |
|---|---|
| `pgs_to_srt.py` | Decode a PGS/`.sup` bitmap subtitle stream and OCR it |
| `fix_ocr.py` | Repair systematic OCR errors; language-aware punctuation |
| `verify_ocr.py` | List remaining OCR suspects with their source bitmaps |
| `chunk_audio.py` | Silence-aligned chunking; writes `cuts.txt` |
| `run_chunks.sh` | Per-chunk WhisperX with resume-safe checkpointing |
| `align_words.py` | Alignment-only pass for word-level timings |
| `build_srt.py` | Rebuild cues with real subtitle constraints |
| `find_suspects.py` | Dictionary check to surface candidate errors |
| `review_pairs.py` | Side-by-side draft vs reference for line-by-line review |
| `asr_ctc.py` | Second-opinion transcript from a wav2vec2 CTC model (optional) |
| `cross_check.py` | Flag words an independent pass did not confirm (low-yield) |
| `adjudicate.py` | Show reference text at a suspect's timecode |
| `apply_fixes.py` | Apply text fixes; asserts timings unchanged |
| `qa_srt.py` | Structural, readability and coverage QA |

## Dependencies

```bash
brew install ffmpeg tesseract tesseract-lang hunspell
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install whisperx
```

Hunspell dictionaries go in `~/Library/Spelling/` (macOS):

```bash
curl -sL -o ~/Library/Spelling/fr.dic \
  https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/dictionaries/fr.dic
curl -sL -o ~/Library/Spelling/fr.aff \
  https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/dictionaries/fr.aff
```
