#!/usr/bin/env python3
"""Second, architecturally independent transcript from NVIDIA Canary-1B-v2 via MLX.

    run_canary.py <audio.wav> <work-dir> <out.srt> <lang> [chunk-seconds]

Canary is a genuinely different family from Whisper (FastConformer encoder +
transformer decoder, NeMo lineage), so its errors are uncorrelated with
Whisper's -- which is the whole reason it earns a place in the pipeline.

Two things this script exists to get right:

1. **Canary has no segmentation of its own.** mlx-audio returns
   `start=0.0, end=0.0` for every segment and its `generate` does no chunking,
   defaulting to `max_tokens=200` -- long audio silently truncates. So the audio
   is split into fixed windows here and each window's timecode is reconstructed
   from its index. Cue boundaries are therefore window boundaries, NOT speech
   boundaries: this file is a *wording* source, never a timing source.

2. **source_lang must equal target_lang.** Canary also does speech translation,
   and if the two differ you get a translation instead of a transcript. Worse,
   mlx-audio's CLI silently drops `--language` (it filters kwargs against the
   named parameters of each model's `generate`, and Canary takes
   `source_lang`/`target_lang`), falling back to en/en -- so French audio comes
   back as fluent English and nothing warns you. This script always sets both
   explicitly, which is why it uses the Python API rather than the CLI.

Resume-safe: each window's text is checkpointed to <work-dir>/NNNN.txt and
re-runs skip completed windows.
"""
import os
import subprocess
import sys
import time

DEFAULT_CHUNK = 30.0
MODEL = 'CogniSoftOrg/canary-1b-v2-mlx-bf16'


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}'


def split_audio(audio, chunk_dir, chunk_s):
    """Fixed-length split. Deliberately not silence-aligned: windows must be a
    known duration so each one's absolute timecode is index * chunk_s."""
    os.makedirs(chunk_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(chunk_dir) if f.endswith('.wav'))
    if existing:
        print(f'reusing {len(existing)} existing windows in {chunk_dir}')
        return [os.path.join(chunk_dir, f) for f in existing]
    subprocess.run(
        ['ffmpeg', '-y', '-v', 'error', '-i', audio, '-f', 'segment',
         '-segment_time', str(chunk_s), '-c', 'copy',
         os.path.join(chunk_dir, 'c%04d.wav')],
        check=True,
    )
    return [os.path.join(chunk_dir, f)
            for f in sorted(os.listdir(chunk_dir)) if f.endswith('.wav')]


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    audio, work, out_srt, lang = sys.argv[1:5]
    chunk_s = float(sys.argv[5]) if len(sys.argv) == 6 else DEFAULT_CHUNK

    from mlx_audio.stt.utils import load_model

    windows = split_audio(audio, os.path.join(work, 'canary_chunks'), chunk_s)
    print(f'{len(windows)} windows of {chunk_s:g}s')

    model = load_model(MODEL)
    texts = []
    computed = 0          # windows actually decoded, so resumes report honest rates
    t0 = time.time()
    for i, wav in enumerate(windows):
        ck = os.path.join(work, 'canary_chunks', f'{i:04d}.txt')
        if os.path.exists(ck):
            texts.append(open(ck, encoding='utf-8').read().strip())
            continue
        computed += 1
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
            rate = (done * chunk_s) / max(time.time() - t0, 1e-6)
            print(f'  {done}/{len(windows)} windows, {rate:.0f}x realtime', flush=True)

    with open(out_srt, 'w', encoding='utf-8') as fh:
        n = 0
        for i, txt in enumerate(texts):
            if not txt:
                continue        # silent/music window: emit nothing, not a blank cue
            n += 1
            fh.write(f'{n}\n{ts(i * chunk_s)} --> {ts((i + 1) * chunk_s)}\n{txt}\n\n')

    print(f'wrote {out_srt}: {n} windows with speech of {len(windows)}')
    if computed:
        elapsed = time.time() - t0
        audio_s = computed * chunk_s
        print(f'{elapsed:.0f}s for {audio_s:.0f}s of new audio '
              f'= {audio_s / max(elapsed, 1e-6):.0f}x realtime')
    else:
        print('all windows were already checkpointed; nothing recomputed')


if __name__ == '__main__':
    main()
