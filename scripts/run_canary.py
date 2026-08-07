#!/usr/bin/env python3
"""Second, architecturally independent transcript from NVIDIA Canary-1B-v2 via MLX.

    run_canary.py <audio.wav> <work-dir> <out.srt> <lang> [target-seconds]

Canary is a genuinely different family from Whisper (FastConformer encoder +
transformer decoder, NeMo lineage), so its errors are uncorrelated with
Whisper's -- which is the whole reason it earns a place in the pipeline.

Three things this script exists to get right:

1. **Canary has no segmentation of its own.** mlx-audio returns
   `start=0.0, end=0.0` for every segment and its `generate` does no chunking,
   defaulting to `max_tokens=200` -- long audio silently truncates. So the audio
   is split into windows here and each window's timecode is reconstructed from
   the actual cut points, not from its index. Cue boundaries are still window
   boundaries, NOT speech boundaries: this file is a *wording* source, never a
   timing source.

2. **Windows are silence-aligned and short (~`target-seconds`, default 8s),
   not fixed 30s blocks.** A 30s window can and does bundle several unrelated
   exchanges into one blob of prose with a single timecode, which makes it easy
   to skim past a mismatch between what Canary heard and what your draft
   contains -- the exact failure mode this script used to invite. Shorter,
   silence-aligned windows can't fully solve that (a fast back-and-forth still
   gets bundled if there's no pause), but they cut it down a lot, and they mean
   a bad window's evidence is easier to line up against `check_swallowed_spans.py`
   or your draft directly. This does cost more model calls than one call per
   30s -- expect roughly 4x the number of windows.

3. **source_lang must equal target_lang.** Canary also does speech translation,
   and if the two differ you get a translation instead of a transcript. Worse,
   mlx-audio's CLI silently drops `--language` (it filters kwargs against the
   named parameters of each model's `generate`, and Canary takes
   `source_lang`/`target_lang`), falling back to en/en -- so French audio comes
   back as fluent English and nothing warns you. This script always sets both
   explicitly, which is why it uses the Python API rather than the CLI.

Resume-safe: the cut plan is written to <work-dir>/canary_chunks/cuts.txt once
and reused, and each window's text is checkpointed to <work-dir>/canary_chunks/NNNN.txt.
"""
import os
import re
import subprocess
import sys
import time

DEFAULT_TARGET = 8.0
MODEL = 'CogniSoftOrg/canary-1b-v2-mlx-bf16'


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}'


def duration(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', path], capture_output=True, text=True).stdout
    return float(out.strip())


def silences(path, noise='-40dB', mind='0.4'):
    r = subprocess.run(
        ['ffmpeg', '-hide_banner', '-i', path, '-af',
         f'silencedetect=noise={noise}:d={mind}', '-f', 'null', '-'],
        capture_output=True, text=True)
    return [float(m) for m in re.findall(r'silence_start:\s*([0-9.]+)', r.stderr)]


def plan_cuts(dur, sil, target):
    """Same planner as chunk_audio.py: snap to a nearby detected silence within
    target/6 of the ideal cut point, else fall back to the fixed interval."""
    cuts = [0.0]
    while cuts[-1] + target < dur - target / 5:
        want = cuts[-1] + target
        window = target / 6
        cand = [s for s in sil if abs(s - want) < window and s > cuts[-1] + target / 2]
        cuts.append(min(cand, key=lambda s: abs(s - want)) if cand else want)
    cuts.append(dur)
    return cuts


def split_audio(audio, chunk_dir, target):
    """Silence-aligned split. Cut points are persisted so a resumed run
    reconstructs the same windows without redetecting silences differently."""
    os.makedirs(chunk_dir, exist_ok=True)
    cuts_path = os.path.join(chunk_dir, 'cuts.txt')
    if os.path.exists(cuts_path):
        cuts = [float(x) for x in open(cuts_path) if x.strip()]
        print(f'reusing {len(cuts) - 1} existing windows from {cuts_path}')
    else:
        dur = duration(audio)
        sil = silences(audio)
        cuts = plan_cuts(dur, sil, target)
        with open(cuts_path, 'w') as f:
            f.write('\n'.join(f'{c:.3f}' for c in cuts))

    windows = []
    for i in range(len(cuts) - 1):
        dst = os.path.join(chunk_dir, f'{i:04d}.wav')
        if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
            subprocess.run(
                ['ffmpeg', '-y', '-v', 'error', '-ss', f'{cuts[i]:.3f}',
                 '-t', f'{cuts[i + 1] - cuts[i]:.3f}', '-i', audio, dst],
                check=True)
        windows.append(dst)
    return windows, cuts


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    audio, work, out_srt, lang = sys.argv[1:5]
    target = float(sys.argv[5]) if len(sys.argv) == 6 else DEFAULT_TARGET

    from mlx_audio.stt.utils import load_model

    windows, cuts = split_audio(audio, os.path.join(work, 'canary_chunks'), target)
    print(f'{len(windows)} windows, target {target:g}s, silence-aligned')

    model = load_model(MODEL)
    texts = []
    computed = 0            # windows actually decoded, so resumes report honest rates
    computed_audio_s = 0.0
    t0 = time.time()
    for i, wav in enumerate(windows):
        ck = os.path.join(work, 'canary_chunks', f'{i:04d}.txt')
        if os.path.exists(ck):
            texts.append(open(ck, encoding='utf-8').read().strip())
            continue
        computed += 1
        computed_audio_s += cuts[i + 1] - cuts[i]
        out = model.generate(
            wav,
            source_lang=lang,      # must equal target_lang -- see docstring
            target_lang=lang,
            max_tokens=512,        # 200 truncates ~30s of dense dialogue
            use_pnc=True,
        )
        txt = out.text.strip()
        with open(ck, 'w', encoding='utf-8') as fh:
            fh.write(txt)
        texts.append(txt)
        if i % 20 == 0:
            done = i + 1
            elapsed_audio = cuts[min(done, len(cuts) - 1)]
            rate = elapsed_audio / max(time.time() - t0, 1e-6)
            print(f'  {done}/{len(windows)} windows, {rate:.0f}x realtime', flush=True)

    with open(out_srt, 'w', encoding='utf-8') as fh:
        n = 0
        for i, txt in enumerate(texts):
            if not txt:
                continue        # silent/music window: emit nothing, not a blank cue
            n += 1
            fh.write(f'{n}\n{ts(cuts[i])} --> {ts(cuts[i + 1])}\n{txt}\n\n')

    print(f'wrote {out_srt}: {n} windows with speech of {len(windows)}')
    if computed:
        elapsed = time.time() - t0
        print(f'{elapsed:.0f}s for {computed_audio_s:.0f}s of new audio '
              f'= {computed_audio_s / max(elapsed, 1e-6):.0f}x realtime')
    else:
        print('all windows were already checkpointed; nothing recomputed')


if __name__ == '__main__':
    main()
