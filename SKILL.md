# Magic Subtitles

Produce a subtitle file that matches **what the voices actually say**, with
frame-accurate timing, for a dubbed film.

The naive approach — run Whisper, ship the SRT — fails three ways: timings
drift, proper nouns get mangled, and long stretches of speech collapse into
unreadable blocks. This skill fixes all three by transcribing the audio three
times with three unrelated model families, forced-aligning the result to word
level, and correcting it against a real subtitle found for the film.

**The four models, one job each:**

| Model | Job |
|---|---|
| `whisper-large-v3-mlx` | Timing (word-level, via forced alignment) + a first wording guess |
| `Canary-1B-v2` | Independent listen, equal weight on wording |
| `Qwen3-ASR-1.7B` | Independent listen, equal weight on wording |
| `wav2vec2-large-xlsr-53-<lang>` | Forced-aligns the chosen words to the audio for per-word timing — never changes a word |

None of the three ASR passes is assumed more accurate than the others. Where
all three agree, a line needs no more thought; where they disagree, that's the
audio telling you it's genuinely ambiguous. This is what buys you the right to
stop scrutinizing 90%+ of the file and put your attention where it matters.

---

## Step 0 — Interview the user

Ask up front, one message:

1. **Language of the dub.** Default French, offer English, accept anything —
   just get the right code into `mlx_whisper --language`, `run_canary.py` /
   `run_qwen3.py`'s language arg, and `align_words.py`'s aligner-model choice.
2. **Do you have a reference subtitle for this release**, or should one be
   searched for online? Always search even if the user has one — a second
   reference resolves cases the first can't. See Step 4.
3. **Deliver just the `.srt`, soft-mux into the container, or hard-burn into
   the picture?** Default is just the `.srt`. Hard-burn is common but always
   comes *after* the text is settled (Step 14) — it's a lossy, permanent
   re-encode, and re-burning because a line changed costs a whole encode.

Always run Whisper `large-v3` on Metal via `mlx-whisper` — never ask, never
offer `turbo`/`medium`/`small`. It's the accurate model, and Metal makes it
~7x realtime, so the old speed/accuracy tradeoff doesn't exist on Apple
Silicon. If Metal is unavailable, fall back to WhisperX on CPU and warn the
user it will take hours instead of minutes; don't chunk a whole film into one
CPU pass (OOM risk — chunk it, same as the Metal path).

---

## Step 1 — Gather context

WebSearch for a plot synopsis, the full cast/character list (note whether
character names change in the dub — most Disney-style dubs keep English
names), place names, song titles in the dub language, and any invented
terminology. Write it to `glossary.md`.

This is the highest-leverage step in the whole pipeline for the effort it
costs. Almost every non-obvious correction later depends on knowing the
plot: a homophone slip reads as a plausible sentence until you know the
antagonist's actual name, or that a character doesn't exist.

**Watch for a wordless vocalise or foreign-language chant** at the very open
of some films — if a reference subtitle has no cues over that stretch, that's
the tell it's intentionally untranslated; don't invent lyrics for it later.

---

## Step 2 — Probe the file

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name,channels:stream_tags=language,title \
  -of default=noprint_wrappers=1 "$MOVIE"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$MOVIE"
df -h .   # writes ~duration x 32kB/s of WAV plus chunks of the same size again
```

Note the audio stream index in the target language and its channel count.

---

## Step 3 — Extract the audio

Center channel only for 5.1/7.1 — dialogue lives there, score and effects sit
wider. Free separation, measurably helps:

```bash
ffmpeg -y -v error -i "$MOVIE" -map 0:$AIDX -vn -af "pan=mono|c0=c2" \
  -ar 16000 -c:a pcm_s16le audio.wav
```

Sanity-check level (`ffmpeg -i audio.wav -af volumedetect -f null -`, expect
mean around −25 to −15 dB). Don't apply cleanup filters or run source
separation (Demucs etc.) — both were tested and made recognition worse or net
hurt more than they helped.

---

## Step 4 — Find a reference subtitle

Search the web for a subtitle file in the **dub language**, for this film —
exact release name first, then title + year. A plain-text SRT/ASS track
already muxed into the file is a free win if one exists (`ffmpeg -i "$MOVIE"
-map 0:$SIDX reference.srt`); a bitmap (PGS/VobSub) track is not worth the OCR
machinery it'd take to extract — skip it and rely on the web search instead.

Save the best result as `web.srt`. It's a **different translation**, not a
transcript of the dub — in a reference run only ~54% of words overlapped the
actual dub wording. That makes it:

- **Strong** on proper-noun spelling and overall scene meaning.
- **Weak** on exact wording — never let it settle a choice between two
  readings that mean the same thing.
- **Useless as a timing source** — it's timed to a different release. Spot
  check dialogue starts at a few points across the runtime; a stable few
  seconds of offset is fine, growing drift means it's not a safe reference.

If nothing is found, say so and proceed with the synopsis/glossary alone —
expect a longer user-review table at the end.

---

## Step 5 — Transcribe three times

Split the audio once, on silence, so no cut lands mid-word. Default is 120s
chunks, not a longer target — see below for why:

```bash
python3 scripts/chunk_audio.py audio.wav work/chunks    # 120s default
```

**Whisper large-v3, Metal:**

```bash
uv tool install mlx-whisper
"$(uv tool dir)/mlx-whisper/bin/python" scripts/run_whisper.py work/chunks work/srt fr
```

Two things matter here, not just tuning:

1. **120s chunks, not ~600s.** Whisper's own segment timestamps come from a
   coarse internal estimate, and a wrong one doesn't get fixed downstream —
   `align_words.py` (Step 6) only refines word timing *inside* whatever
   window Whisper proposed; it can't correct the window itself. Measured on a
   real film: the error does NOT grow with position inside a long chunk
   (within-chunk correlation ~0, so it isn't cumulative drift), but shorter
   chunks still shrink the worst-case tail substantially (P90 |timestamp
   error| dropped from 9.6s to 7.1s in that test, mean/median improved too).
   Treat 120s as the default, not a per-run decision.
2. **Load the model once, don't shell out per chunk.** `run_whisper.py` loops
   in one process — the alternative (invoking the `mlx_whisper` CLI once per
   chunk) reloads the whole model every time, which is fine at ~11 chunks and
   expensive at ~50. This is what keeps 120s chunking free instead of ~5x
   slower.
3. **The anti-hallucination settings still matter.** Without
   `condition_on_previous_text=False` and `hallucination_silence_threshold=2`,
   `mlx-whisper` (no built-in VAD) falls into repetition loops — check the
   combined output for repeated lines:
   `grep -vE '^[0-9]+$|-->|^$' work/srt/*.srt | sort | uniq -c | sort -rn | head`

**Canary-1B-v2 and Qwen3-ASR-1.7B, independent second and third opinions:**

```bash
uv tool install mlx-audio
"$(uv tool dir)/mlx-audio/bin/python" scripts/run_canary.py audio.wav work canary.srt fr
"$(uv tool dir)/mlx-audio/bin/python" scripts/run_qwen3.py  audio.wav qwen3.srt fr
```

Both window the audio into short (~8s), silence-aligned chunks and reconstruct
timing from the real cut points — mlx-audio's raw output has no per-window
timestamps. Their cue bounds are **windows, not real cue timing** — treat them
as wording sources only, never for timing. A run of the same phrase repeated
many times inside one window is a decoder loop, not the dub repeating itself —
ignore that window's wording, don't "fix" it as a single line.

Run these three passes sequentially (each wants the whole GPU); ~45 minutes
total for a 2-hour film on Metal.

---

## Step 6 — Word-level alignment

```bash
python3 scripts/align_words.py work/chunks work/srt work/words.json fr
```

`align_words.py` needs a WhisperX venv (`uv venv --python 3.12 .venv && uv pip
install whisperx`), not MLX — it's a separate CTC forward pass over the
Whisper segments, recovering per-word timing that whisper's own SRT export
throws away. Each output word also carries a `seg_start`/`seg_end` tag — the
specific whisperx-internal output window it came from, not just its position
in the original segment list (whisperx.align() can split one input segment
into several internally). Step 7's VAD cross-check depends on this tag to
group words back into their true source window, so keep it if you ever touch
this script's output format.

Clamping and duration-suspect flagging (`clamp_durations.py`) now happens in
Step 7, **after** the VAD cross-check, not directly on this raw output —
keep reading before running it.

---

## Step 7 — Re-anchor words across a real pause (VAD cross-check)

wav2vec2 forced alignment (Step 6) sometimes places a word or short phrase
several seconds before it's actually spoken. Root cause, confirmed against
real data: the aligner front-loads a window's recognized words toward
whichever edge is unconstrained, whenever that window contains a real
stretch of silence next to the actual speech. Two distinct signatures of this
are independently detectable and correctable against a separate, audio-only
VAD pass — see `vad_reanchor.py`'s docstring for the full derivation;
summary:

- **Detector A — leading silence absorbed into the first word.** The word's
  own raw duration balloons (already flagged by `clamp_durations.py`'s
  suspects mechanism) because nothing before it bounds its start. Its END is
  usually right (anchored by real speech right after it) — only the START is
  wrong.
- **Detector B — trailing dead air left unclaimed.** No single word's
  duration looks wrong, but the whole recognized cluster sits compressed near
  the window's start, leaving a multi-second gap between the last word's end
  and the *original Whisper segment's* declared end that no word accounts
  for. Checked only on the last whisperx output window belonging to that
  original segment — matched by time overlap, exact-containment-first, never
  by position (adjacent back-to-back segments caused wrong matches under a
  naive tolerance-based match; fixed).

```bash
uv pip install silero-vad   # CPU-only, no GPU needed; shares the whisperx venv

python3 scripts/detect_vad_regions.py audio.wav work/vad_regions.json
python3 scripts/vad_reanchor.py work/words.json work/srt work/chunks/cuts.txt \
  work/vad_regions.json work/words_reanchored.json
```

`detect_vad_regions.py` runs Silero VAD once over the whole film's
`audio.wav` and only needs Step 3's output — it doesn't depend on
transcription or alignment at all, so it can run any time after Step 3, in
parallel with Steps 5/6 if useful. It reads the WAV directly with Python's
`wave` module rather than silero_vad's own `read_audio` — that needs a
torchaudio sox backend not installed in this environment.

`vad_reanchor.py` groups words by their `seg_start`/`seg_end` tag rather than
by position, then applies both detectors and writes a re-anchored copy. It
must run **before** `clamp_durations.py` — Detector A's ratio check needs the
raw, unclamped durations, so the clamp step now takes the reanchored file as
input:

```bash
python3 scripts/clamp_durations.py work/words_reanchored.json work/chunks/cuts.txt \
  work/words_clamped.json work/suspects.json
```

Keep `work/words_reanchored.json` (unclamped) and `work/words_clamped.json`
(clamped) as separate files, not the same path — overwriting the input
discards the raw pre-clamp durations `suspects.json` already points at,
which you'll want on hand if a flagged span needs a closer look later.

`clamp_durations.py` fixes the *timing* damage from any word whose duration
blew up (a stuck decoder loop, or a long segment around a short utterance),
and flags anything that was over 6s **before** clamping into
`work/suspects.json`. A flagged word is not resolved by clamping — it was
that long for a reason, and the reason might be that real dialogue is hiding
under it. For every flagged span:

```bash
python3 scripts/check_swallowed_spans.py work/suspects.json draft.srt \
  CAN=canary.srt QWN=qwen3.srt WEB=web.srt
```

This prints what every source says happened in that span. If an independent
source shows several real sentences where your draft has one word or none,
that's swallowed dialogue — treat the whole span as a reconstruction case in
Step 9, not a single-word fix. (`draft.srt` doesn't exist yet the first time
through — build it in Step 8, then come back and run this check before Pass B.)

**Validated on a real feature-length run**, cross-checked against known
ground truth: corrected timestamps matched an ear-check exactly, word count
before/after was identical (this only nudges existing words' timing, never
re-aligns, so nothing is lost or duplicated), and `qa_srt.py` passed before
and after with no overlaps introduced.

**Known limitations:**

1. **Long, un-split repeated-phrase clusters are unreliable for Detector B.**
   whisperx doesn't always split a repeated phrase into separate output
   windows — a phrase repeated several times in a row with no strong
   internal punctuation can stay as one giant window, while the same phrase
   broken up by exclamation marks gets split into separate windows
   (punctuation gives the aligner an internal boundary). Detector B shifts
   a whole cluster as
   one rigid block, preserving internal spacing; on a long un-split cluster
   that shift can overshoot the window's own end and collide with unrelated
   adjacent content. Treat a Detector B correction on an unusually long
   cluster as lower-confidence, and visually sanity-check any shift that
   looks implausibly large before trusting it — same "don't auto-apply the
   big ones blindly" spirit as the rest of this skill's correction steps.
2. **A third failure mode exists that neither detector catches.** A Whisper
   segment can be cold-start-anchored to exactly its own chunk's t=0.000 (no
   prior context) while still showing a *tight* internal window — words fill
   it almost completely, so there's no leading-duration anomaly (A) and no
   trailing dead air (B) to flag. Hit example: a short line opening a new
   chunk had its declared window filled almost exactly by its own words, yet
   the true speech was a few seconds later — the VAD region it actually
   belonged to started a few seconds *before* the chunk boundary,
   already claimed by the tail of the previous chunk's own (correctly
   reanchored) content. Not automated; still needs a human catching it by
   ear, same as before this mechanism existed. A future "Detector C" would
   check whether a chunk-start-anchored segment's declared start actually
   coincides with where VAD says speech begins, accounting for the preceding
   chunk possibly already claiming that region.

---

## Step 8 — Build the draft

```bash
python3 scripts/build_srt.py work/words_clamped.json work/chunks/cuts.txt draft.srt fr
```

Rebuilds cues from acoustic word onsets under real subtitle constraints: max
6s, max 84 chars over 2 lines, splits on silences and sentence-final
punctuation, extends fast cues into trailing silence to ease reading speed,
never overlaps.

---

## Step 9 — Correct the transcript

This is the step that separates a usable file from a rough one. A dictionary
check alone only catches errors that spell as non-words — most real ASR
errors are valid words in the wrong place.

### Pass A — dictionary sweep

```bash
python3 scripts/find_suspects.py draft.srt fr --foreign en
python3 scripts/adjudicate.py draft.srt web.srt fr WORD1 WORD2 ...
```

Cheap, catches the obvious (mangled proper nouns, foreign-language bleed).

### Pass B — three-source line-by-line review

Run this as a single `Workflow` call, not manual batches — a script computes
the batch list from the real cue count, so full-file coverage follows from
the loop bound instead of being tracked by hand. ~150 cues per batch is a
good size (8-12 batches for a 90-120 min film).

```js
export const meta = { name: 'pass-b-review', description: 'Three-source review', phases: [{title:'Review'}] }
const DIR = '/path/to/working/dir'
const TOTAL_CUES = 1240   // the REAL count from `grep -c '\-\->' draft.srt` --
                          // hardcode it in the script body. Passing it through
                          // Workflow's `args` has been unreliable in practice
                          // (args.totalCues arrived undefined mid-session);
                          // don't spend time debugging that, just hardcode it.
const BATCH = 150
const batches = []
for (let lo = 1; lo <= TOTAL_CUES; lo += BATCH) batches.push([lo, Math.min(lo+BATCH-1, TOTAL_CUES)])

const FINDINGS_SCHEMA = { type:'object', properties: { findings: { type:'array', items: { type:'object',
  properties: {
    cue:{type:'integer'}, timestamp:{type:'string'}, verdict:{type:'string', enum:['fix','ask','dismiss']},
    action:{type:'string', enum:['text_fix','retime','drop','reconstruct']},
    fix_text:{type:'string'}, new_start:{type:'string'}, new_end:{type:'string'},
    replaces_cues:{type:'array', items:{type:'integer'}}, justification:{type:'string'},
  }, required:['cue','timestamp','verdict'] } } } }

const METHOD = `...paste the Authority Table and Decision Procedure below, plus the
  glossary/synopsis content, plus any known suspects.json spans, into every batch's
  prompt -- a fresh subagent has none of this context otherwise...`

const results = await pipeline(batches, ([lo,hi]) => agent(
  `Review cues ${lo}-${hi} of draft.srt against CAN=canary.srt, QWN=qwen3.srt, ` +
  `WEB=web.srt. Read every cue in range, don't sample. ${METHOD}`,
  { schema: FINDINGS_SCHEMA, label: `cues ${lo}-${hi}` }))

return { findings: results.filter(Boolean).flatMap(r => r.findings || []) }
```

**If a batch errors** (a content-filter false positive has happened on
otherwise-ordinary content — combat scenes, loud exclamations), don't debug
the Workflow: just re-run that one batch's prompt through a direct `Agent`
call and merge its findings in by hand.

**Authority table:**

| Source | Authority on | NOT authority on |
|---|---|---|
| Draft (Whisper) | Timing, and the wording being judged | — |
| CAN (Canary) | What was said — independent listen | Timing (windows) |
| QWN (Qwen3-ASR) | What was said — independent listen | Timing (windows) |
| WEB (reference) | Proper-noun spelling, scene meaning | Exact wording, timing |

**Decision procedure, in order:**

1. All three ASRs agree → almost certainly right, even if WEB words it
   differently (translation divergence, not an error). This is most of the
   file — the payoff for running three models is being able to stop thinking
   about these.
2. Two agree, one differs → majority is usually right, but check the odd one
   out against WEB/glossary before dismissing.
3. All three differ → genuinely ambiguous; use WEB + glossary + synopsis to
   pick the best-fitting reading, or verdict `ask` if nothing fits confidently.
4. All three agree but the line is nonsense in context → only WEB/glossary/
   synopsis can save it; `ask` rather than invent if nothing does.
5. A proper noun is involved → WEB/glossary decide spelling, always — a
   character's name never changes spelling from the original just because a
   dub keeps it, so a phonetic ASR variant of a glossary name is a misheard
   word, not an alternate form.

**A cue every source agrees on can still be wrong.** All three ASRs can
independently converge on a fluent-sounding sentence that just isn't what was
said — `J'ai bien évité mon surnom` (I successfully *avoided* my nickname)
read fine to all three passes, but the line is `J'ai bien mérité mon surnom`
(I've *earned* my nickname) — nonsensical-in-context beats fluent-in-isolation
as a signal, and only a human catching it while watching resolved this one.
Expect a few of these to survive even a careful Pass B; that's what Step 12 is
for.

**A systematic name mishearing can hide from batch review entirely.** If a
character's dub-kept name sounds like an unrelated real word in the dub
language, the mishearing can recur across dozens of separate cues, scattered
across different Pass B batches, every one reading as perfectly fluent text
in isolation — no single batch has enough context to flag it as wrong, each
occurrence looks locally fine. After Pass B, grep the whole file for
every glossary name and its phonetically-plausible near-misses in the dub
language; fix any hit with a single global find-and-replace.

Check each hit's context before sweeping it, though — a character being
briefly misnamed can be the joke, not an ASR error (confirm against WEB/the
synopsis). A blanket find-and-replace erases that along with the real
mishearings.

### Pass C — hallucination sweep

```bash
grep -inE "sous-titrage|subtitle|merci d'avoir regardé|abonnez-vous" draft.srt
grep -vE '^[0-9]+$|-->|^$' draft.srt | sort | uniq -c | sort -rn | head
```

Any subtitling-house credit or a line repeated many times at regular
intervals (often the end-credits tail, once real dialogue has ended) is a
hallucination — confirm against the web reference's last real cue, then drop
the whole tail.

### Applying

```bash
python3 scripts/apply_findings.py draft.srt findings.json corrected.srt asks.json
```

Aggregate every Pass B batch's findings plus the Pass A/C fixes into one
`findings.json` before applying (see the script's docstring for the schema —
it handles single-cue text/retime/drop and multi-cue hallucination-loop
reconstruction, and splits `ask`-verdict findings into `asks.json` for
Step 12). It renumbers cues on output.

**Re-run `qa_srt.py` (Step 13) immediately after applying.** A `retime` can
overlap an untouched neighbouring cue that wasn't part of the fix — this is
expected, not a bug in the script, and needs a manual timing nudge on
whichever side is at fault.

---

## Step 10 — Repair sentence boundaries

```bash
python3 scripts/fix_sentence_breaks.py corrected.srt fr --report \
  --glossary "Name1,Name2,..." --words work/words_clamped.json
```

Rebuilding cues from word onsets loses whisper's own sentence-final
punctuation at the seams, so a new sentence can start mid-line with no break.
Pass a full glossary name list — a naive check fires on every proper noun.

Once satisfied, apply for real with `--out fixed.srt`, adding `--song-ranges
"MM:SS-MM:SS,..."` for every musical number (find the ranges by grepping the
draft for a couple of distinctive lyric words). Inside those ranges the
script only inserts line breaks, never a full stop — sung lines conventionally
carry no terminal punctuation.

Re-run `--report` on the output; it should converge to 0 genuine hits within
a pass or two (any stragglers are usually right at a song-range boundary —
widen the range slightly and re-check).

---

## Step 11 — Punctuation sweep

Regrouping words into cues (Step 8) and Pass A/B/C's wording review don't
catch every punctuation problem — missing or wrong terminal marks, missing
commas (direct address, clauses joined by "mais"/"et"), stray/orphaned marks
left over from a cue split, inconsistent ellipsis (`…` vs `...`).

Batch it the same way as Pass B (Step 9): one agent per ~140-cue range,
reading every cue in range, flagging only clear-cut issues — never a
stylistic preference. Skip song/lyric passages entirely; they carry no
punctuation by convention (same exemption as Step 10). If a batch errors on
ordinary content, that's the Pass B content-filter false positive — just
re-run that one batch.

Split findings by confidence before touching anything:

- **High confidence** (mechanical — a stray doubled mark, a missing terminal
  period on an otherwise-complete sentence, an ellipsis format fix): apply
  directly.
- **Low/medium confidence** (a passage has so little punctuation left that
  fixing it means guessing sentence breaks, not inserting a mark): don't
  guess. Present the cue numbers, timestamps, and current text to the user
  for an ear-check, same as any other `ask`-verdict item from Step 12 — a
  wrong guess here changes what the line says, not just how it's punctuated.

---

## Step 12 — Hand the remainder to the user

Present every `ask`-verdict item from `asks.json` as one table, sorted by
timestamp, with a best guess where you have one and the sources that produced
it. Never invent a line to avoid asking — a flagged cue the user can check by
ear beats a confident wrong one they won't think to.

Expect several rounds: the user resolves a batch, and their corrections
sometimes reveal a fix Pass B made confidently but wrong (see the "every
source agreed and it was still wrong" note in Step 9) — when that happens,
show your work: quote what each of the three sources actually said at that
timestamp, so the user can judge whether your read of the evidence, not just
the conclusion, was reasonable.

---

## Step 13 — QA

```bash
python3 scripts/qa_srt.py fixed.srt web.srt
```

Checks overlaps, non-monotonic or zero-length cues, line/length limits,
reading-speed distribution, and coverage against the reference (cues in
`web.srt` with nothing nearby in the output). Investigate any gap over a few
seconds — cross-check `CAN`/`QWN`/`WEB` at that timestamp before concluding
it's really missing rather than just differently worded or split.

**Separate on-screen signage from missed dialogue** before trusting a
coverage number — a reference track often subtitles forced narrative (signs,
captions), usually ALL CAPS, with no spoken counterpart at all.

---

## Step 14 — Deliver

Name the file to match the video (`Movie.Name.2024.1080p.fr.srt`) so players
auto-load it.

**Every video this skill writes for playback carries AAC-LC stereo audio.**
Most releases ship AC3/E-AC3 5.1, which Android TV/Chromecast silently drop —
picture with no sound, no ffmpeg warning either. Never `-c:a copy` into a
delivered video:

```bash
ffmpeg -y -i "in.mkv" -map 0:v:0 -map 0:a:0 -c:v copy \
  -c:a aac -profile:a aac_low -b:a 192k -ac 2 "out.mkv"
```

Soft-mux (picture untouched, subtitles toggleable):

```bash
ffmpeg -y -i "$MOVIE" -i fixed.srt -map 0:v:0 -map 0:a:0 -map 1 \
  -c:v copy -c:a aac -profile:a aac_low -b:a 192k -ac 2 \
  -c:s srt -metadata:s:s:0 language=fra "output.mkv"
```

### Hard burn — lossy and permanent, do it last

Confirm ffmpeg actually has libass first — Homebrew's core `ffmpeg` formula no
longer bundles it, so `subtitles`/`ass` filters silently don't exist and the
failure looks like a quoting problem, not a missing filter:

```bash
ffmpeg -filters | grep -E '\bsubtitles\b|\bass\b'   # empty -> no libass
ls /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg          # keg-only, check before installing
```

**`FontSize` in `force_style` is not pixels** — it's in the coordinate space
of the ASS script (ffmpeg synthesizes `PlayResX 384 / PlayResY 288` from an
SRT), so the same number renders at wildly different pixel sizes depending on
source resolution. Convert to ASS and rewrite `PlayRes` to the real frame size
first, then size in real pixels:

```bash
ffmpeg -y -i fixed.srt work/base.ass
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$MOVIE")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$MOVIE")
sed -i '' -e "s/^PlayResX: .*/PlayResX: $W/" -e "s/^PlayResY: .*/PlayResY: $H/" work/base.ass
```

Then, relative to the real frame height: **font ≈ 6-7% of frame height**
(72px at 1080p is a good default — noticeably larger than the ~5% textbook
cinema norm, and worth it on a living-room TV), **outline ≈ 6% of the font**,
**MarginV ≈ 4-5% of frame height**, bold on. Check letterboxing first
(`cropdetect`) — a scope film may have a black bar wide enough that a larger
font never touches the picture at all; render one still per candidate size
and measure the actual text bounding box before deciding, rather than
eyeballing it.

```bash
ffmpeg -y -i "$MOVIE" -map 0:v:0 -map 0:a:0 -sn -vf "ass=work/base.ass" \
  -c:v libx264 -crf 18 -preset medium \
  -c:a aac -profile:a aac_low -b:a 192k -ac 2 "output-burned.mkv"
```

Hardware encoders (`*_videotoolbox`) run ~9x realtime vs. roughly realtime for
`libx264 -preset medium`, at a quality cost — ask the user's tolerance and
whether they want a size budget; if a hardware encoder only takes a bitrate
(not CRF), probe ~90s of actual encode and extrapolate rather than guessing
one.

Verify the finished file's duration matches the source and the audio track is
`aac,LC,2,48000` before handing it over — never overwrite the original.

---

## Pitfalls

- **Container corruption can truncate a sequential read silently** — ffmpeg
  exits 0 with a short file. Compare the container's duration against the
  extracted WAV's before trusting either.
- **A word list for foreign-language detection must never include ambiguous
  tokens** — `a`/`on`/`or`/`in` all exist as real French words too and will
  delete genuine dialogue that happens to score as "English."
- **A regex fix pattern must allow newlines** where the original had a
  literal space — cue text wraps across lines, so `"wrong word"` silently
  fails to match `"wrong\nword"`. Always `\s+`.

---

## Scripts

| Script | Purpose |
|---|---|
| `chunk_audio.py` | Silence-aligned chunking for the Whisper pass; writes `cuts.txt` |
| `run_whisper.py` | Whisper large-v3 pass, model loaded once, looped over all chunks |
| `align_words.py` | wav2vec2 forced alignment for word-level timing |
| `detect_vad_regions.py` | Silero VAD pass over the whole film; absolute-time speech regions |
| `vad_reanchor.py` | Cross-checks word timings against VAD regions; fixes words placed across a real pause |
| `clamp_durations.py` | Clamp implausible word durations; flags spans that may hide swallowed dialogue |
| `check_swallowed_spans.py` | Diff every source's content against the draft for a flagged span |
| `build_srt.py` | Rebuild cues from word timings under real subtitle constraints |
| `find_suspects.py` | Dictionary check to surface candidate errors (Pass A) |
| `adjudicate.py` | Show the reference text at a suspect word's timecode |
| `run_canary.py` | Independent second transcript, Canary-1B-v2 on Metal |
| `run_qwen3.py` | Independent third transcript, Qwen3-ASR-1.7B on Metal |
| `fix_sentence_breaks.py` | Find and repair unpunctuated sentence starts; song-aware |
| `apply_findings.py` | Apply Pass B/C findings (fix/retime/drop/reconstruct); splits off `ask` items |
| `qa_srt.py` | Structural, readability, and coverage QA |

## Adapting to another language

The pipeline is language-generic; the language code needs to go in the same
handful of places every time:

| Where | Note |
|---|---|
| `run_whisper.py` language arg | ISO-639-1 (`fr`, `en`, ...) |
| `run_canary.py` / `run_qwen3.py` language arg | ISO-639-1; Canary covers 25 European languages, Qwen3-ASR covers 52 |
| `align_words.py` language arg | picks the wav2vec2 model |
| `build_srt.py` / `fix_sentence_breaks.py` language arg | punctuation-spacing rules (e.g. French needs a space before `! ? ; :`) |

## Dependencies

```bash
brew install ffmpeg
uv tool install mlx-whisper      # Step 5, Metal
uv tool install mlx-audio        # Step 5, Canary + Qwen3-ASR
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install whisperx   # Step 6, wav2vec2
uv pip install silero-vad   # Step 7, VAD cross-check; CPU-only, shares the whisperx venv
```

`mlx-audio` installs its own interpreter and serves both Canary and Qwen3-ASR
— call scripts with it directly:
`"$(uv tool dir)/mlx-audio/bin/python" scripts/run_canary.py ...`.

`brew install ffmpeg` alone is not enough for hard-burn (no libass); see
Step 14 for the keg-only `ffmpeg-full` check.

Both MLX packages need Apple Silicon; each model downloads a few GB on first
use.
