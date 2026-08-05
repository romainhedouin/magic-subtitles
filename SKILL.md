---
name: magic-subtitles
description: Generate subtitles that match a film's dubbed audio, transcribing twice with two independent model families (Whisper large-v3 on Metal and NVIDIA Canary-1B-v2) plus forced alignment, then correcting the result against target-language reference subtitles. Use when a user wants accurate subtitles for a movie, wants subtitles matching what the voices actually say (rather than a translation of the original script), or wants to fix badly-timed or mistranslated subtitles.
---

# Magic Subtitles

Produce a subtitle file that matches **what the voices actually say**, with
frame-accurate timing.

The naive approach — run Whisper, write the SRT — fails in three predictable
ways: timings drift by seconds, proper nouns are mangled, and long runs of
audio collapse into unreadable blocks. This skill fixes all three.

**About the examples in this document.** Worked examples throughout come from
one reference run — a French-dubbed 1998 animated feature — and are written as
`what the ASR produced` → `what it should be` → *how that was determined*. They
illustrate the *shape* of each problem, not facts about that film. Expect the
same shapes in any film and language: invented names, homophone swaps, a fixed
idiom broken, an insult heard as a proper noun.

The core idea: **transcribe the audio twice with two unrelated model families,
then correct the result against human subtitles in the same language.** Whisper
`large-v3` (on Metal) supplies the timing and the primary wording; NVIDIA
Canary-1B-v2 is an independent second listener, so where the two agree a line
needs no further thought and where they differ you have found the exact places
the audio is ambiguous. The subtitle files — the track inside the movie file, plus
one fetched online — are human translations timecoded to a real release; they tell
you what a scene *means* and how every proper noun is *spelled*.

Four sources, each authoritative about something different, none authoritative
about everything. Step 4 sets out which is which, and Pass B in Step 8 is where
they are read against each other line by line.

---

## Step 0 — Interview the user

Ask these three questions **before doing any work**. Use the AskUserQuestion
tool, all three in a single call.

### Q1. Which Whisper model?

**Run `large-v3` on Metal via `mlx-whisper`, not on CPU via CTranslate2.** Same
weights, same accuracy, measured **7× the speed** — this is the default now, and
Q1 exists only to let the user pick a *smaller* model, not a different runtime.
Times below are for a ~90-minute film on an M4 Air; see the measured table after
Q3 for the numbers behind this.

| Model | Metal (`mlx-whisper`) | CPU (CTranslate2) | Notes |
|---|---|---|---|
| `large-v3` | **~12 min** | 1–2 h | Best accuracy. **Use this.** |
| `large-v3-turbo` | ~5 min | 20–40 min | Distilled decoder. Measurably worse on names and rare words. |
| `medium` | ~5 min | 20–40 min | Weaker again. |
| `small` | ~3 min | 10–20 min | Draft quality only. |

Because Metal makes `large-v3` cheap, the old speed-vs-accuracy trade is mostly
gone: there is now no good reason to accept a worse model to save ten minutes.

**Never use `turbo`.** Its distilled decoder gives up accuracy precisely on proper
nouns and rare vocabulary — the words that matter most and are hardest to spot
afterwards (measured on the reference run, same audio: `les eux` where `large-v3`
read `les Huns`). On Metal it saves about seven minutes, so there is nothing to
weigh.

### Measured speeds — use these to give the user an estimate

**All figures measured on an M4 Air (16 GB), 16 kHz mono WAV, French audio.**
To estimate: `wall seconds ≈ audio seconds ÷ multiple`. A 2 h film is 7200 s of
audio, so ÷7 ≈ 17 min, ÷15 ≈ 8 min, ÷33 ≈ 3.5 min, ÷1 ≈ 2 h. The Parakeet
multiple held within 2% across two different films, so treat these as reliable.
Add a **one-time ~5 min model download** per model, and ~10 s for ffmpeg audio
extraction.

| Model | Runtime | × realtime | 2 h film | Own timestamps? |
|---|---|---|---|---|
| `large-v3` | **`mlx-whisper` (Metal)** | **7.3×** | **~16 min** | Yes, segment-level |
| `large-v3-french` | `mlx-whisper` (Metal) | 7.1× | ~17 min | Yes, but 69 cues >20 s |
| `large-v3` | WhisperX (CPU) | ~0.8–1.5× | 1.5–3 h | Yes, segment-level |
| Canary-1B-v2 | `mlx-audio` (Metal) | **15×** | ~8 min | **No — all zeros** |
| Parakeet TDT 0.6B v3 | `parakeet-mlx` (Metal) | 33× | ~3.5 min | Yes, incl. word-level |

The 7.3× for `mlx-whisper` **depends on the anti-loop flags** in Step 5a. Without
them it measures 5.2× *and* produces garbage — see there. `faster-whisper`/WhisperX
only load Whisper-architecture weights; other models need their own MLX runtime,
but the **forced-alignment stage is model-agnostic**, so any ASR can feed it.

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

**Canary is the accuracy pick, Parakeet the speed pick.** Canary is clearly better
French, but `mlx-audio` returns `start=0.0, end=0.0` for every segment — it has no
segmentation at all, so it is only usable *with* the alignment stage, and the audio
must be pre-chunked with ffmpeg (its `generate` has no chunking and defaults to
`max_tokens=200`, which truncates anything longer than ~30 s of speech).

Two `mlx-audio` traps:

- **It silently drops `--language`.** The CLI filters kwargs against the named
  parameters of each model's `generate`, and Canary takes `source_lang`/`target_lang`.
  So `--language fr` vanishes, the `en`/`en` defaults apply, and Canary returns an
  English **translation** of French audio. Always pass
  `--gen-kwargs '{"source_lang":"fr","target_lang":"fr"}'`, or use the Python API.
  Source and target must always be equal — this pipeline never wants translation.
- **Cohere-transcribe-03-2026 does not currently work on MLX**, despite topping the
  Open ASR leaderboard (5.42% WER). Both `beshkenadze` conversions (fp16 and 8-bit)
  emit token soup that is identical across different audio and equally broken in
  English, `mlx-community/…-mlx-8bit` is an empty repo, and `cohere_asr`'s VAD path
  crashes (`mx.array.astype(np.float32)`). Do not spend time on it; the untried
  routes are the `openasr` Rust binary and the CoreML/ONNX conversions.

### Q2. Which language is the audio in?

Default **French**. Offer **English** as the other suggestion. Accept anything
else the user types — the pipeline is language-generic, but you need the right
code in four places: Whisper (`fr`), Canary `source_lang`/`target_lang` (`fr`),
tesseract (`fra`), hunspell (`fr`).

Canary covers 25 European languages. If the user names something outside that
set, run Step 5a only and tell them the second-opinion pass is unavailable for
this language.

### Q2b. Do you have a target-language subtitle file for this release?

Ask it here, not later. Step 4b needs one and will go searching online if the
user has none — but they often have a `.srt` sitting next to the video, and
knowing that up front saves a search. Discovering it *after* transcribing wastes
the verification stage.

### Q3. Burn the subtitles into the video?

Three outcomes, and the user should understand the trade:

- **No (default)** — just the `.srt` file. Players load it automatically when
  it sits beside the video with a matching filename.
- **Soft-mux** — remux the subtitle track into the container. Fast, picture
  untouched, toggleable in the player.
- **Hard burn (pixel-embed)** — subtitles painted into the picture. Requires a
  full video re-encode: lossy, permanent, and the file cannot be turned off.
  Roughly realtime with `libx264 -preset medium`, but ~9× realtime with a
  hardware encoder.

**Hard burn is a normal, common end state — expect to do it.** What it is not is
the *next* step: because the re-encode is lossy and permanent, it comes after the
user has worked through the uncertainties you flagged in Step 8 and is happy with
the subtitle file. So take the answer here as a statement of intent, deliver the
`.srt` (or soft-mux) first, and burn once the text is settled. Re-burning because
a line changed costs a whole re-encode.

When you do reach the burn, three further answers change the command. Get them
**then**, not now, since they depend on the finished subtitle file and on probing
the video:

- **font size**, as a fraction of frame height, and whether two-line cues may
  overlap the picture when the source is letterboxed
- **encoder** — hardware (minutes, weaker per bit) vs software (hours, best
  quality per byte)
- **size budget**, which is how you calibrate a hardware encoder's bitrate

See "Hard burn" in Step 10 for why the font size cannot be a fixed number.

These three are the *minimum*. Ask more whenever an answer would change the
plan — see "The endgame is the user" in Step 8. Checking
before a 1–2 hour transcription costs the user a moment; discovering the wrong
assumption afterwards costs both of you the run.

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
tells you that `les uns` is really `les Huns` (a homophone, and the synopsis
names the Huns as the antagonists), or that `défaire Léa` is `défaire les Huns`
(the ASR invented a person; the character list has no such name). Without the synopsis
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

## Step 4 — Get the reference sources

This is the step that makes the result reliable. **Do not skip it.**

Gather up to three text sources before transcribing. They do different jobs and
are **not** interchangeable — this table is the one to keep in mind for the rest
of the run:

| Source | Label | What it is good for | What it must never do |
|---|---|---|---|
| Subtitle track inside the movie file (4a) | `REF` | Detecting that a line is wrong; recovering meaning | Bulk-replacing wording — it is a different translation |
| Target-language subtitle found online (4b) | `WEB` | **Proper-noun spellings**, scene meaning, second opinion on `REF` | Settling wording, or supplying timings |
| Canary transcript (Step 5b) | `CAN` | Confirming or challenging the actual words heard | Providing timings — its cues are fixed windows |

Both text sources must be in the **dub/target language** (typically French), never
the original English. Be clear-eyed about what they are, though: an ordinary
subtitle file is virtually always translated from the English original rather than
transcribed from the dub, so it is a *semantic parallel text* — right about what
the scene means, unreliable about the exact words the voices say. An SDH track is
the exception and the prize (see 4b).

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

**Required, unless the user supplied one — always search.** Do not treat this as
optional and do not skip it because 4a succeeded. A second reference resolves
cases where the first is too divergent to judge, which is the single biggest
cause of errors that survive to the end. Save it as `web.srt`.

Search by exact release name first, then by title and year
(`"Intouchables 2011 sous-titres français srt"`). Check the release folder and
the container's own tracks before searching the web — a `.srt` is often sitting
right next to the video file.

It must be **in the dub/target language** (typically French). Do **not** settle
for an English file: the original-language script disagrees with the dub
everywhere and generates confident false corrections.

Best target: a **SDH / "sourds et malentendants" / hearing-impaired** subtitle
track. Unlike an ordinary subtitle translation, SDH transcribes the *dub audio*
itself, so it matches what the voices say almost word for word. That is the one
artefact that would resolve nearly everything a normal reference cannot. Search
for it explicitly.

**Understand what you have found, because it decides how much to trust it.**
Ordinary target-language subtitles are *virtually always translated from the
English original*, not transcribed from the dub. So they carry the meaning of the
scene faithfully and the wording of the dub only loosely — they diverge from the
dub in the same places, and for the same reason, as the in-file track. Two such
files are two translations of one English script, not two witnesses to the audio.

What that means in practice, and it is worth being precise because this is where
false corrections come from:

- **Strong on proper nouns.** Names survive translation intact, so this is the
  most reliable thing a web subtitle gives you. ASR spells names phonetically
  (`Salut, chienne peau` for `Chien-Po`); the subtitle has the canonical spelling
  of every character, place and brand.
- **Strong on meaning.** When the two ASR passes produce plausible-but-different
  French, the subtitle line usually makes it obvious which reading fits the scene.
- **Weak on wording.** It cannot settle a choice between two French readings that
  mean the same thing. Do not let it.
- **Useless as timing.** It is timed to *some* release, not necessarily yours. Use
  it as a **drift check** (does dialogue start at about the same time at 10 min,
  60 min, 100 min?) and never as a timing source. See "Validate timing by drift,
  not by offset" in Step 9.

Two rules:

- Use it to **verify and adjudicate**, never to bulk-replace the transcript.
  Do not copy a full screenplay into the output file; quote only what you need
  to settle a specific word.
- If both 4a and 4b produced a file, keep them **separate** (`reference.srt` and
  `web.srt`) and pass both to Pass B. They agree often, and where they disagree
  you have learnt something real about which one tracks the dub.

**If you genuinely cannot find one, say so and carry on.** The in-file track plus
the Step 1 synopsis is enough; just expect a longer manual-review table at the end.

### 4c. Fallback: ask the user

If the file has no subtitle track in the target language and you found nothing
online, ask the user to supply a reference subtitle file for this release. A
`.srt` they already have works fine.

If there is no reference at all, you can still transcribe — but say clearly that
verification was limited to the synopsis, grammar and internal consistency, and
expect materially more errors to survive.

**Ask this before transcribing, not after.** It is the single decision that most
changes the quality of the result, and the user may have a suitable file to hand.
Discovering it after a 1–2 hour run wastes the verification stage entirely.

### An important caveat to tell the user

The subtitle track and the dub are usually **different translations**. In this
skill's reference run, only ~54% of words overlapped, and the median per-cue
match was 20%. Same meaning, different wording.

So the reference is **not** ground truth to copy from. It is a *semantic
parallel text*: use it to work out what a garbled word must have been, never to
overwrite Whisper's wording wholesale. The dub transcript is what matches the
voices; that is the whole point.

---

## Step 5 — Transcribe twice, with two different model families

**Run both passes. This is not optional.** Whisper supplies the timing and the
primary wording; Canary is an architecturally independent second listener, and
where the two agree a line needs no further thought. Together they cost about
25 minutes for a 2-hour film on Metal, which is less than the old single CPU pass.

There is an **optional third pass** (Step 5c), French audio only. It is a Whisper
fine-tune, so it is *not* a third independent vote — it earns its place by
transcribing speech the other two drop, not by breaking ties.

Split the audio once, on silence, so no cut lands mid-word — both passes reuse it:

```bash
python3 scripts/chunk_audio.py audio.wav work/chunks 600   # ~10-min targets
```

### 5a. Whisper `large-v3` on Metal

```bash
uv tool install mlx-whisper
```

```bash
mlx_whisper work/chunks/000.wav \
  --model mlx-community/whisper-large-v3-mlx --language fr \
  --condition-on-previous-text False --hallucination-silence-threshold 2 \
  --output-format srt --output-dir work/srt --output-name 000
```

**The two flags are mandatory, not tuning.** `mlx-whisper` has no VAD, so with
default settings it falls into repetition loops that WhisperX's VAD would have
prevented. Measured on a 10-minute French clip:

| | default | `--condition-on-previous-text False --hallucination-silence-threshold 2` |
|---|---|---|
| Cues produced | 297 | 173 |
| Cues that were one looped line | **217 (73%)** | 0 |
| Speed | 5.2× realtime | **7.3× realtime** |

The clean run is also the faster run, because the loop was burning decode time
generating `C'est plus prudent !` 217 times. Still run Pass C afterwards — this
same clip yielded `Sous-titrage ST' 501`, a subtitling-house hallucination.

Chunking remains worthwhile even though Metal is fast: it gives per-chunk
checkpointing, and `align_words.py` in Step 6 consumes per-chunk SRTs.

**If Metal is unavailable** (non-Apple hardware, or MLX broken), fall back to
WhisperX on CPU with `bash scripts/run_chunks.sh work/chunks work/srt large-v3 fr`
and warn the user it will take 1.5–3 h instead of ~16 min. `run_chunks.sh` skips
any chunk whose SRT already exists — in the reference run a process kill destroyed
the wrapper but ~70 minutes of completed work survived and resumed cleanly. On CPU,
**do not run a full-length film in one pass**: it loads the whole waveform as
float32 (~5.6 GB for 90 min) alongside the model, VAD and alignment nets, and gets
OOM-killed on a 16 GB machine. The kill surfaces as exit code **137**, and a shell
wrapper can mask it as "exit 0" — check for the output file, not the exit status.

Run either pass in the background and monitor completions rather than blocking.

### 5b. Canary-1B-v2, the independent second pass

```bash
uv tool install mlx-audio
```

```bash
"$(uv tool dir)/mlx-audio/bin/python" scripts/run_canary.py \
  audio.wav work canary.srt fr
```

`run_canary.py` handles the three things Canary gets wrong on its own, and its
docstring explains each: it splits the audio into 30 s windows (Canary does no
chunking and silently truncates at `max_tokens=200`), reconstructs a timecode per
window (mlx-audio returns `start=0.0, end=0.0` for everything), and always sets
`source_lang == target_lang` (otherwise you get a *translation*, and mlx-audio's
CLI silently drops `--language`, so French audio comes back as fluent English with
no warning). It checkpoints per window and resumes.

**`canary.srt` is a wording source, not a timing source.** Its cue boundaries are
30-second window boundaries, so never let it near the timing stage. Its
characteristic failure is decoder repetition inside a window
(`Escorte. Escorte. Escorte.`) — repetition in `CAN` means "ignore this window",
not "the dub repeats itself".

### 5c. `large-v3-french`, the coverage pass (French audio, optional)

```bash
bash scripts/setup_whisper_fr.sh          # one-time, ~10 min, builds the MLX model
```

```bash
mlx_whisper audio.wav \
  --model "$HOME/.cache/whisper-mlx/whisper-large-v3-french" --language fr \
  --condition-on-previous-text False --hallucination-silence-threshold 2 \
  --output-format srt --output-dir . --output-name french
```

(`--output-name` takes the stem, not the filename — it appends `.srt` itself.)

`bofenghuang/whisper-large-v3-french` is a full fine-tune of `large-v3` (32
layers, `n_vocab` 51866 — not a distil variant). Same runtime, same flags, same
~17 min for a 2-hour film as Step 5a. Run it on the **whole** `audio.wav`, not the
chunks: its output never reaches the alignment stage, so per-chunk SRTs buy
nothing here, and the anti-loop flags are still mandatory — it loops worse than
`large-v3`, not better.

**Read this before deciding to run it.** Measured head-to-head against Step 5a on
the same French film (*Intouchables*, 1 h 52 m, identical audio and flags):

| | `large-v3` | `large-v3-french` |
|---|---|---|
| Words transcribed | 10 225 | **10 635** |
| Speech duration covered | 74.2 min | **83.3 min** |
| Empty `...` filler cues | 83 | **0** |
| `Abonnez vous` hallucinations | 5 | **1** |
| Cues stuck in a repetition loop | **5** | 16 |
| Worst loop | 7× | **112×** (`Ah ! Ah ! Ah !…`) |
| Cues over 20 s | **35** | 69 |
| Cues total | 1890 | 964 |

Only 39% of aligned blocks came back identical, so it disagrees with Step 5a
constantly — 2 761 words changed across the 424 differing blocks.

**What it is good for.** It recovers whole exchanges that `large-v3` threw away as
`...` filler — the single largest win, and exactly the class of miss Pass B is
expensive to catch by hand:

| Time | `large-v3` | `large-v3-french` |
|---|---|---|
| 00:06:59 | `... ...` Ça va aller ? | On l'a prévenu, y a un brancard qui arrive dans une 2nde. On vous laisse là, ça va aller ? |
| 01:17:45 | Ah d'accord. D'accord, y'a pas de soucis. `...` | Allô, Dries. Qu'est-ce que vous faites ? Je vous dérange ? … Vous voulez vous barrer, c'est ça ? |

It is also better on French mechanics: `faut qu'on voie` not `faut qu'on voit`,
`s'il n'est pas` not `s'il est pas`, `6 mois` not `Six mois`, `kilomètres` not `km`.

**What disqualifies it as a primary pass.** It loops far harder than `large-v3`
*with the identical anti-loop flags* — 112 consecutive `Ah !` swallowed a real
exchange about phantom pain at 00:42:09, and at 01:23:38 an entire scene came back
as the single cue `Sous-titrage MFP.` It also emits half as many cues covering
*more* speech, so 69 cues run over 20 s — the same mega-cue failure that got
Parakeet rejected above.

So: **`french.srt` is a wording and coverage source, never a timing source, and
never an independence vote.** It shares `large-v3`'s weights, so when it agrees
with Step 5a that agreement is worth much less than Canary agreeing — see the
independence rule under "On running several models". Pass it to Pass B as `FRW`,
and treat a `FRW`-only line the way you treat any single-source claim: as a lead
to verify, not a correction to apply.

**Skip it** when the audio is not French — there is no equivalent fine-tune wired
up for other languages, and pointing this step at non-French audio just gives you
`large-v3` with extra steps.

---

## Step 6 — Recover word-level timings

WhisperX aligns to word level internally but **throws that away** when writing
SRT, leaving whisper's coarse segment boundaries. Recover it with an
alignment-only pass — a wav2vec2 forward pass, no re-transcription:

```bash
python3 scripts/align_words.py work/chunks work/srt work/words.json fr
```

### Clamp implausible word durations before building cues

The aligner can only place words **inside the segment window whisper gave it**.
Wherever whisper closed a segment early, the words get crammed, and wherever it
left a long window around a short utterance, one word can absorb seconds of
silence. Check the distribution before trusting it:

```bash
python3 - <<'EOF'
import json
d = json.load(open('work/words.json'))
durs = sorted(w['end'] - w['start'] for ws in d.values() for w in ws)
n = len(durs)
for q in (50, 90, 99, 99.9):
    print(f'p{q}: {durs[int(n * q / 100)]:.2f}s')
print('max:', round(durs[-1], 2), ' over 2s:', sum(1 for x in durs if x > 2))
EOF
```

A healthy French/English distribution sits around p50 ≈ 0.15 s and p99 ≈ 1 s. A
maximum in the double digits means an outlier, and it does **two** kinds of
damage — the obvious over-long cue, and a much less obvious one: because
`build_srt.py` splits whenever the gap between consecutive words exceeds 0.8 s,
an inflated word manufactures a fake gap and **tears its sentence into three
cues**. Fixing the SRT afterwards would leave those bad splits in place.

Clamp at the word level instead, keeping each word's start (its onset is
usually right; only the end is wrong):

```python
cap = min(4.5, max(1.2, 0.12 * len(word.strip())))   # generous for sung/held notes
w['end'] = min(w['end'], w['start'] + cap)
```

In the reference run this touched 0.6% of words and simultaneously removed every
cue over 6 s and rejoined the torn sentences.

**Two things measured not to work** — don't spend time rediscovering them:

- **Widening the segment windows before aligning** (padding each into the gap to
  its neighbours) sounds like the principled fix and does nothing when whisper's
  segments are already back-to-back, because `min(next_start, …)` leaves no room.
  It made the output worse: cues over 6 s went 4 → 10 as isolated words sprawled.
- **A single word's bad duration is not evidence the alignment failed.** Verify
  against the audio before "fixing" a pause: an energy profile of the window will
  often show the silence is real, meaning the split is correct and only the
  duration needs clamping.

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

### Pass B — contextual line-by-line read against all sources (mandatory)

Read **every cue** against **every source** at the same timecode. Do not skip this
and do not sample it. Work in batches of 100–150 cues so each batch fits
comfortably in context:

```bash
python3 scripts/review_pairs.py draft.srt 1 150 \
  CAN=canary.srt FRW=french.srt REF=reference.srt WEB=web.srt
python3 scripts/review_pairs.py draft.srt 151 300 \
  CAN=canary.srt FRW=french.srt REF=reference.srt WEB=web.srt
# ... continue to the end; each run prints the next batch's command
```

Every source you obtained must be passed. Omit `REF` if the film had no in-file
track, `WEB` if the search genuinely failed, or `FRW` if you skipped Step 5c, but
never omit `CAN`.

**How to weigh the voices.** Each answers a different question, and confusing them
is how false corrections get made:

| | Authority on | Explicitly NOT authority on |
|---|---|---|
| `ASR` (Whisper) | **Timing**, and the wording being judged | — |
| `CAN` (Canary) | What was **said** — an independent listen | Timing (30 s windows) |
| `FRW` (large-v3-french) | **Speech the others missed**; French mechanics | Timing (mega-cues); independence |
| `REF` (in-file subtitle) | **Whether** a line is wrong; meaning | Exact wording |
| `WEB` (online subtitle) | **Proper-noun spellings**; scene meaning | Any wording choice; timing |

`REF` and `WEB` are **not** independent of each other — both are typically
translations of the same English script — so their agreement is weak evidence
about the dub. Neither is `FRW` independent of `ASR`: it is a fine-tune of the
same weights, so `ASR`+`FRW` agreement is close to one witness speaking twice.
Only `ASR` and `CAN` are genuine independent witnesses to the audio.

**Where `FRW` changes the procedure:** it does not get a vote in steps 1–3 below.
Use it in one place only — when `ASR` shows `...`, a suspiciously short line, or
nothing at a timecode where the film clearly has dialogue, check `FRW` for the
words. That is the case it measurably wins (+9 minutes of recovered speech). When
`FRW` supplies a line no other source has, verify it against `REF`/`WEB` for
meaning before accepting: it is also the source most prone to inventing a
112-times repetition or a `Sous-titrage MFP.` where a whole scene should be.

Read them as a decision procedure, in this order:

1. **`ASR` and `CAN` agree** → the line is almost certainly right. Two independent
   architectures rarely share an error. Move on, even if `REF` words it differently
   — that is just translation divergence. This is what makes the two-model cost
   worth it: it lets you *stop thinking* about most cues.
2. **`ASR` and `CAN` differ** → the audio is genuinely ambiguous here, and this is
   the highest-yield signal in the whole pipeline. Now use `REF` and `WEB` for the
   meaning of the scene, plus the Step 1 synopsis and glossary, and pick the
   reading that fits. Prefer whichever ASR is consistent with the established
   context; Canary was measurably the better French transcriber, so on a pure
   coin-flip it wins — but only after the meaning check, never before.
3. **Both ASR passes agree but the line is nonsense** in context → the audio
   misled both. Only `REF`/`WEB`/synopsis can save it, and often nothing can: send
   it to the user-review table rather than inventing a fix.
4. **A proper noun is involved** → `WEB`/`REF` decide the spelling, always,
   regardless of what the two ASR passes heard.

Then, for each cue, ask:

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

#### Verify every pattern landed, with newlines normalised

`apply_fixes.py` reports the cues it *changed*, never the patterns that matched
nothing. So a fix can silently fail and the run still looks successful. Write
every literal space in a pattern as `\s+` by default, and then prove it:

```python
import json, re
cfg = json.load(open('fixes.json'))
# Join each cue's lines into ONE space-separated string before searching.
cues = []
for b in re.split(r'\n\s*\n', open('final.srt', encoding='utf-8').read().strip()):
    L = b.split('\n')
    if len(L) >= 3:
        cues.append(' '.join(' '.join(L[2:]).split()))
missed = [p for p, _ in cfg['replacements'] if re.search(p, '\n'.join(cues), re.M)]
print(f'patterns still matching the output: {len(missed)}', missed)
```

**The newline normalisation is the whole point.** Searching the output with its
line breaks intact reproduces the very failure you are hunting: a pattern with a
literal space does not match the wrapped phrase in the output either, so it is
reported as absent and you conclude the fix worked. In the reference run that
false "all clean" hid **16 failed fixes across three rounds**, each asserted to
the user as applied. Normalising newlines first exposed all of them at once.

A pattern can never bridge a **cue** boundary — `apply_fixes.py` works per cue.
When a sentence runs across two cues, the words either side of the split live in
different strings, so no amount of `\s+` will join them:

```
cue N     "... and he said the old"      <- pattern "the old CAR" cannot match
cue N+1   "CAR was finished."               across these two cues
```

Anchor the pattern to the cue that actually contains the wrong word
(`^CAR\s+was finished`). Watch for this whenever a fix refuses to land even
after `\s+` has been applied.

Re-run the check after *every* build. Line-break changes can move a phrase in or
out of a single line, which changes whether a pattern matches at all.

### Judgement rules

- **Only fix what you can justify** from the reference, the synopsis, or a fixed
  idiom. If the translations diverge too far to tell, leave the word alone. A
  wrong word the user can spot beats a confident-sounding invention they cannot
  audit.
- Musical numbers diverge most between dub and subtitle translations, so that is
  where you will resolve least. Expect it.

### On running several models for "multiple sources"

The pipeline now runs two (Step 5), but the constraint that governs *which* two is
narrow, and the failures below are all still real. What matters is that the second
model be **independent and individually strong**; either property missing and it
adds nothing:

- **Two Whisper variants are not independent.** They share weights and training
  data, so they reproduce each other's errors. Confirmed directly: whisper.cpp
  `large-v3` independently produced `Chifou` for a character named `Chi Fu` —
  the identical error the CTranslate2 `large-v3` run made, which had already
  been corrected by hand. A second Whisper pass
  will confidently agree with the first that `les Huns` is `les uns`.
- **An independent but weak model adds nothing.** A wav2vec2 CTC second pass is
  architecturally unrelated to Whisper and found **0 new errors** while flagging a
  third of all cues, because a model with no language model disagrees with Whisper
  nearly everywhere — and when disagreement is the base rate, disagreement carries
  no information. Validated against 14 known errors: it caught 7 and ranked none
  of them in its top 50. Do not rebuild it.
- **Canary-1B-v2 is the one that satisfies both** — NeMo FastConformer lineage
  (genuinely unrelated to Whisper) *and* strong French. It is the "individually
  strong, genuinely diverse second system" this section used to list as untested.
  Measured against Parakeet TDT v3 on the same French film, Canary read
  `Ça y est, les voilà` and `100 euros que je les mets dans le vent` where Parakeet
  read `Dégage toi les voilà` and `100 euros que les mettre dans le nom`.
- **A same-family fine-tune buys coverage, not independence.**
  `bofenghuang/whisper-large-v3-french` (Step 5c) is a `large-v3` fine-tune, so the
  rule above applies in full: its agreement with Step 5a is nearly worthless as
  corroboration. What it does buy, measured on the same French film, is **+410
  words and +9 minutes of speech** that `large-v3` emitted as `...` filler, plus
  correct French mechanics (`faut qu'on voie`, `6 mois`, `kilomètres`). It pays for
  itself as a *coverage* source and must never be scored as a third vote. Its cost
  is real: 16 looping cues against `large-v3`'s 5, one of them 112× `Ah !`, and 69
  cues over 20 s. Third pass, optional, French only.
- **Do not run them concurrently on CPU.** CTranslate2 already saturates the
  cores; parallel runs are proportionally slower each with no wall-clock gain.
  On Metal this no longer applies, but run them sequentially anyway — each pass
  wants the whole GPU.

An honest caveat on the evidence: Canary was compared head-to-head against
Parakeet, and against Whisper only on sample lines, not by WER against a human
reference. Its *independence* from Whisper is architectural fact; its *superiority*
to Whisper on French is not established. That is why Step 5a stays the timing and
primary-wording source and Canary is the second opinion, rather than the reverse.

Spend the effort on Step 1 (synopsis) and Pass B (line-by-line) first. Both
were measured to catch far more, for far less compute.

### Validate any detector before you trust it

The transferable lesson. Before believing a new error-detection technique,
**run it against errors you have already confirmed** and measure how many it
finds and how highly it ranks them. This turns "seems useful" into a number,
and it is the only thing that separates a real check from a plausible-looking
list. It cost one command here and prevented shipping a 234-cue list of noise
as though it were diligence.

### The endgame is the user, and that is by design

Once the sources are exhausted, what remains are the lines where the dub says
something no other text records — and on the reference run **every correction
after that point came from the user listening.** A question is cheaper than a
wrong guess and cheaper than an hour of compute that finds nothing. Treat it as
the design of the process, not an admission of failure.

- **Never invent a word to avoid asking.** A confident-looking wrong line is
  worse than a flagged one, because the user will not think to check it.
- **Never run more compute purely to avoid asking.** On the reference run an
  extra cross-check pass cost ~30 minutes and found nothing that four listening
  questions had not already settled.
- **Ask early when the answer changes the plan**, not only at the end: no
  reference track (ask *before* transcribing — they may have a subtitle file);
  vague reports of "imperfections" (wrong words, timing and line breaks have
  entirely different fixes).
- **Expect several rounds.** Hand over a table → the user answers a few → apply
  → the answers reveal adjacent errors → hand over a smaller table. This ran
  four times on the reference run. Do not present the first table as final.
- **Apply answers verbatim.** The user heard the audio; you did not. If their
  wording implies an adjacent cue is also wrong, ask — do not silently extend it.

Present the remainder as one batched table, sorted by timestamp so the user can
work through the film linearly, with `HH:MM:SS` they can type into a player and a
**Context** column saying what happens in the scene — that is usually what lets
them recognise the line without hunting. Group by ease: short isolated shouts
first, mid-sentence errors next, sung lines last.

| Time | Whisper heard | Reference says | Context |
|---|---|---|---|
| 00:28:05 | `un vrai purson` | *Torride, non ?* | Boasting after a fire trick — was `pur-sang` |
| 00:46:38 | `Kenny ! McKay !` | *Gengis Khanasson...* | Naming the horse; likely a joke name |
| 00:56:13 | `Monkey !` | *(no reference cue)* | Shouted during the battle |

**Do not tell the user which items to skip.** Marking a few as "probably a
deliberate dub invention, don't waste time" is unreliable in exactly the cases
where it feels safest: on the reference run three items were confidently set
aside as intentional malapropisms and **all three were wrong** — each a
*different* malapropism from the one the ASR produced, which no amount of reading
could have revealed. The user resolved all three in seconds once simply listed.
Offer a hypothesis as a suggested-reading column, never as a reason not to check.

**Expect your own confident fixes to be wrong in a specific direction.**
Corrections are safe when they restore a fixed idiom or a glossary name, and much
weaker when the reference merely *implies* a meaning — the dub may render the same
joke in entirely different words, and a plausible reconstruction reads fine while
being wrong. The recurring shapes, all of them ordinary: a real word heard as a
non-word (`purson` → `pur-sang`, `refiant` → `rufian`), an insult heard as a name
(`Blorbeck` → `blanc-bec`), a synonym in the reference (`Petite maladroite` for
`Petite empotée`), or the dub inventing its own insult (`porc aigre doux` where
the reference has *nouille molle*). Prefer flagging over reconstructing whenever
the evidence is semantic rather than lexical.

A precise 10-line table is worth more to the user than another hour of processing
that finds nothing.

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

### Validate timing by drift, not by offset

Matching output cues to the reference by word overlap and reading the
start-delta distribution is the cheapest real check on timing. Read it correctly:

- **A stable offset is not an error.** Cues built from acoustic word onsets will
  sit consistently *later* than a professional reference, because subtitlers cue
  in slightly before speech. A steady median of roughly +0.3 to +0.6 s is the
  expected signature of correct timing.
- **Drift is the failure to hunt.** Bucket the deltas by position in the film and
  compare medians. Values that stay flat across the whole runtime mean the chunk
  offsets are being applied correctly; a median that grows means a chunking or
  offset bug and nothing downstream can be trusted.

### Separate on-screen signage from missed dialogue

Reference subtitle tracks translate **forced narrative** — signs, captions,
on-screen text — usually in ALL CAPS. Those have no counterpart in a transcript
of the spoken dub, so counting them as "uncovered" understates coverage and
sends you hunting for dialogue that was never spoken. Partition before quoting a
number:

```python
def is_sign(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.85
```

What remains is genuinely missed speech, and it is dominated by very short crowd
shouts and interjections — the hardest thing for the ASR under music and noise.

**A targeted re-transcription of gap windows is worth one attempt and rarely
more.** In the reference run, re-running the largest gaps recovered exactly one
substantive line out of dozens; the rest were genuinely masked. Try it once,
report what it found, and put the remainder in the user-review table rather than
grinding.

Also resist the tempting diagnosis. "Crowd lines are missing because we took the
centre channel and crowds are mixed into the surrounds" is plausible, cheap to
test, and was **measured false** — the centre channel was equally loud or louder
at four of five sampled misses. Compare levels before believing it:

```bash
for t in <timestamps>; do
  ffmpeg -hide_banner -ss $t -t 3 -i "$MOVIE" -map 0:a:0 \
    -af "pan=mono|c0=c2,volumedetect" -f null - 2>&1 | grep mean_volume
done
```

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

### Hard burn (do it once the subtitle text is settled — lossy and permanent)

**First check that ffmpeg can render subtitles at all.** Both the `subtitles`
and `ass` filters come from libass, and plenty of builds ship without it —
Homebrew's core `ffmpeg` formula no longer bundles libass, so a stock `brew
install ffmpeg` cannot burn subtitles. The failure is badly disguised:

```
[AVFilterGraph] No option name near 'final.srt'
Error opening output file out.png.
Error opening output files: Invalid argument
```

That reads like a path, quoting or output-format problem, and you can lose a
long time escaping quotes differently. It only means the filter does not exist.
Check first, and check for a fuller build before installing anything:

```bash
ffmpeg -filters | grep -E '\bsubtitles\b|\bass\b'     # empty  -> no libass
ls /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg           # keg-only, usually has it
/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg -buildconf | grep enable-libass
```

#### `FontSize` is not pixels — size it from the frame height

This is the single most time-consuming trap in the burn step. `FontSize` in
`force_style` is expressed in the coordinate system of the ASS *script*, not the
video. libass scales the script to the frame, so:

```
effective_px  =  FontSize x (frame_height / PlayResY)
```

When ffmpeg converts an SRT it synthesises an ASS header with **PlayResX 384,
PlayResY 288** and `Fontsize 16`. So the same `FontSize` number means wildly
different things depending on the source:

| Frame height | scale vs PlayResY 288 | `FontSize=24` renders as |
|---|---|---|
| 720p  | x2.5  | ~60 px |
| 1080p | x3.75 | **~90 px** |
| 4K    | x7.5  | ~180 px |

A bare `FontSize=24` is therefore meaningless on its own, and on 1080p it is
roughly double a normal subtitle. **Do not carry a fixed number between films.**

The fix is to stop working in the synthetic 288-line space. Convert to ASS,
rewrite `PlayRes` to the real frame size, and then the number *is* pixels:

```bash
ffmpeg -y -i final.srt work/base.ass
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of csv=p=0 "$MOVIE")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$MOVIE")
sed -i '' -e "s/^PlayResX: .*/PlayResX: $W/" -e "s/^PlayResY: .*/PlayResY: $H/" work/base.ass
```

Then size relative to the frame, since that is what actually scales:

- **font ≈ 4.5–5% of frame height** — 1080p → 49–54 px, 720p → 32–36 px, 4K → 97–108 px
- **outline ≈ 6% of the font** (~3 px at 1080p). The ASS default `Outline 1` was
  chosen for a 288-line script; left alone it is a hairline that disappears
  against bright scenes.
- **MarginV ≈ 4–5% of frame height** (~50 px at 1080p). The default `10` sits
  almost on the frame edge once PlayResY is the real height.
- Bold (`-1` in the style's Bold field) reads better over moving picture.

```
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1.5,2,60,60,54,1
                        ^font                                       ^bold                      ^outline    ^marginV
```

Burn the ASS (not the SRT), so the styling is explicit and reproducible:

```bash
ffmpeg -y -i "$MOVIE" -map 0:v:0 -map 0:a:0 -sn \
  -vf "ass=work/base.ass" \
  -c:v libx264 -crf 18 -preset medium \
  -c:a aac -profile:a aac_low -b:a 192k -ac 2 "output-burned.mkv"
```

#### Check for letterboxing before choosing the size

A scope film in a 16:9 container has black bars, and subtitles can often sit
entirely inside the lower bar so they never cover picture at all. Whether they
fit is arithmetic, not taste:

```bash
ffmpeg -hide_banner -ss 600 -i "$MOVIE" -vf cropdetect=round=2 -frames:v 40 -f null - 2>&1 \
  | grep -oE 'crop=[0-9:]+' | sort | uniq -c | sort -rn | head -3
# crop=W:H:X:Y  ->  bottom bar height = frame_height - (H + Y)
```

Whether a two-line cue clears the picture is predictable. Measured across three
renders on a 1080-high frame, agreeing to within 1.6 px:

```
visible ink of a two-line cue  ~=  1.6 x font_px
top edge of that cue           ~=  frame_height - MarginV - 1.8 x font_px
```

Compare that top edge against `crop_h + crop_y` from `cropdetect`. If it lands
above the boundary, the first line sits over the picture, and the shortfall in
pixels is exactly `(crop_h + crop_y) - top_edge`.

Note this is *ink* extent, not line boxes — reasoning from nominal line height
(~1.2 x font per line, so ~2.4 x for two) overestimates by half a line and will
talk you out of a size that actually fits.

Offer the user the choice explicitly, with the pixel figures: it is a real
trade-off between legibility and never obscuring the image, and only they can
weigh it. Expect them to accept a few pixels of overlap for a more readable size.

**Measure the result, do not eyeball it.** Render one still per candidate size
and find the text's actual bounding box:

```bash
ffmpeg -y -v error -copyts -ss <t> -i "$MOVIE" -vf "ass=work/base.ass" \
  -frames:v 1 -update 1 frame.png      # -update 1 is required for a single PNG
```

```python
from PIL import Image
im = Image.open('frame.png').convert('L'); w, h = im.size; px = im.load()
# scan only the lower strip, and require a run of bright pixels -- a single
# threshold over the whole frame will happily "find" highlights in the picture
rows = [y for y in range(int(h * 0.8), h)
        if sum(1 for x in range(0, w, 2) if px[x, y] > 200) >= 8]
print(f'text occupies y={min(rows)}-{max(rows)}')
```

Compare that against the `cropdetect` boundary to state plainly whether a
two-line cue clears the picture, and by how many pixels.

#### Pick the encoder deliberately, and calibrate the bitrate

Hardware encoders (`h264_videotoolbox`, `hevc_videotoolbox`, NVENC, QSV) turn a
multi-hour software encode into minutes — measured at ~9x realtime versus
roughly realtime for `libx264 -preset medium`. They are weaker per bit, and the
right compensation is bitrate headroom, so ask the user for a size budget.

Many hardware encoders expose no CRF/quality mode, only bitrate. Do not guess
the bitrate needed to hit a size target: **run the encode for ~90 seconds, read
the real rate, and solve.** Output is far below the requested `-b:v` on easy
content, and the shortfall is not proportional, so one probe is not enough —
take two and interpolate:

```bash
ffmpeg ... -progress work/encode.progress "$OUT"    # then:
tail -c 500 work/encode.progress | tr '\r' '\n' | grep -E 'out_time=|total_size'
# projected_bytes = total_size x (duration / out_time)
```

Cheap to redo at 9x realtime, so calibrate rather than accept a wrong size.
Always verify the finished file:

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT"   # must equal the source
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

**Container corruption silently truncating a sequential read.** A damaged MKV can
make the demuxer give up partway through — it stops, ffmpeg exits **0**, and you
get a short file with no warning beyond an easily-ignored `invalid as first byte
of an EBML number`. Non-empty is not enough; the file looks perfectly fine.

Never trust a duration you did not compare:

```bash
ffprobe -v error -show_entries format=duration -of csv=p=0 "$MOVIE"   # container
ffprobe -v error -show_entries format=duration -of csv=p=0 audio.wav  # extraction
```

Seeking usually reads straight past the bad point, so recover the tail with
`-ss <truncation_point>`, concatenate it onto the good part, and **prove the
splice is time-accurate before building anything on it** — every cue after the
join depends on it. Compare energy envelopes and look for the lag; do not compare
MD5s, which differ for a harmless reason (the decoder primes differently after a
seek) and will send you chasing nothing.

The same trap applies at the far end: **after a hard burn, check the output
duration against the source.** The stall is not deterministic — the same file may
truncate an extraction and encode fully minutes later — so verify every time.

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
["wrong word here", "right word here"]        # WRONG: misses "wrong\nword here"
["wrong\\s+word\\s+here", "right word here"]  # right
```

Write `\s+` for **every** literal space by default. It costs nothing (it matches
a single space too) and removes the whole failure class in one pass. Retro-fitting
this to an existing fixes file is a one-line transform:

```python
pattern = pattern.replace(' ', '\\s+')
```

This fails *silently* — `apply_fixes.py` reports the cues it did change, not the
patterns that matched nothing. Grepping the output for the old text is **not**
sufficient to catch it, because the old text is wrapped there too and your grep
misses it for the same reason the fix did. See "Verify every pattern landed, with
newlines normalised" in Step 8 for the check that actually works, and note that
patterns cannot cross a *cue* boundary at all.

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
| `review_pairs.py` | Draft vs every source (`CAN`/`FRW`/`REF`/`EN`) for line-by-line review |
| `run_canary.py` | Independent second transcript from Canary-1B-v2 on Metal |
| `setup_whisper_fr.sh` | One-time build of the `large-v3-french` MLX model (Step 5c) |
| `adjudicate.py` | Show reference text at a suspect's timecode |
| `apply_fixes.py` | Apply text fixes; asserts timings unchanged |
| `qa_srt.py` | Structural, readability and coverage QA |

## Adapting to another language

The pipeline is language-generic, but the language appears in several places and
a mismatch fails quietly rather than loudly. Change all of these together:

| Where | French | English | Note |
|---|---|---|---|
| `mlx_whisper --language` | `fr` | `en` | ISO-639-1 |
| `run_canary.py` 4th arg | `fr` | `en` | sets `source_lang` **and** `target_lang` |
| `tesseract -l` (OCR) | `fra` | `eng` | ISO-639-2; needs `tesseract-lang` |
| `hunspell -d` | `fr` | `en_US` | dictionary must be installed |
| `align_words.py` | `fr` | `en` | picks the wav2vec2 alignment model |
| `build_srt.py` / `fix_ocr.py` | `fr` | `en` | punctuation spacing |

Two things that are genuinely language-specific, not just a code:

- **Punctuation spacing.** French puts a space before `! ? ; :`; English does
  not. `SPACE_BEFORE` in `build_srt.py` and `fix_ocr.py` holds the set of
  languages that do — add yours if needed.
- **Stopword, determiner and sentence-boundary logic** in the scripts is French.
  For another language, replace it or the heuristics will misfire. It is small
  and self-contained.

Everything else — chunking, alignment, cue building, the QA checks — is
language-neutral.

## Dependencies

```bash
brew install ffmpeg tesseract tesseract-lang hunspell
uv tool install mlx-whisper     # Step 5a, Metal
uv tool install mlx-audio       # Step 5b, Canary-1B-v2
bash scripts/setup_whisper_fr.sh  # Step 5c, optional, French only (~10 min, 3 GB)
# CPU fallback only, if MLX/Metal is unavailable:
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install whisperx
```

`mlx-audio` installs its own interpreter; call `run_canary.py` with it rather than
the project venv: `"$(uv tool dir)/mlx-audio/bin/python" scripts/run_canary.py ...`.
Both MLX packages need Apple Silicon. Each model downloads ~2–3 GB on first use.
`setup_whisper_fr.sh` is only needed for Step 5c; it builds into
`~/.cache/whisper-mlx/` once and is a no-op on re-run. There is no published MLX
build of that model, so the script converts it — and works around a HuggingFace
Xet transfer that stalls on this repo and a weights-filename mismatch. Read its
header before changing how it downloads.
`align_words.py` (Step 6) still needs the whisperx venv — it uses wav2vec2, not MLX.

**`brew install ffmpeg` is not enough for hard burn.** The core formula no longer
bundles libass, so the `subtitles` and `ass` filters are absent and burn-in is
impossible with that binary — everything else in this skill works fine. Confirm
what you have before promising a burn:

```bash
ffmpeg -filters | grep -E '\bsubtitles\b|\bass\b'    # empty -> cannot burn
```

`ffmpeg-full` carries libass, fontconfig and freetype. It is **keg-only**, so it
does not shadow `ffmpeg` on `PATH` and may already be installed without you
noticing — check before installing anything, then call it by full path:

```bash
ls /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
FF=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
```

Hunspell dictionaries go in `~/Library/Spelling/` (macOS):

```bash
curl -sL -o ~/Library/Spelling/fr.dic \
  https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/dictionaries/fr.dic
curl -sL -o ~/Library/Spelling/fr.aff \
  https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/dictionaries/fr.aff
```
