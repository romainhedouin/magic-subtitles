#!/usr/bin/env python3
"""Third transcript from Qwen3-ASR-1.7B via MLX -- independent of BOTH Whisper
and Canary.

    run_qwen3.py <audio.wav> <out.srt> <lang> [chunk-seconds]

Qwen3-ASR is a separate lineage again (Qwen omni-style encoder + LLM decoder),
so its errors correlate with neither Whisper's nor Canary's. Three witnesses
that disagree independently is what makes "all three agree -> stop thinking
about this cue" a valid shortcut.

Two things this script exists to get right:

1. **No segmentation of its own.** `generate` returns ONE segment per call with
   `start=0.0, end=<clip length>` -- there are no internal timestamps, exactly
   like Canary. So the audio is split into fixed windows here and each window's
   timecode is reconstructed from its index. Cue boundaries are therefore window
   boundaries, NOT speech boundaries: this file is a *wording* source, never a
   timing source.

2. **`max_tokens` defaults to 8192 and that is a live footgun.** A 30 s window
   holds ~100 tokens of speech; when the decoder loops it will happily generate
   all 8192, and one window then takes MINUTES instead of ~2 s. Measured: a
   single window burned 5 minutes before being capped. `max_tokens=220` plus a
   repetition penalty removed every loop from an 81-minute film (0 looping
   windows, vs 3 for Canary on the same audio).

Resume-safe: each window's text is checkpointed to <out-dir>/qwen3_ck/NNNN.txt
and re-runs skip completed windows.
"""
import os
import subprocess
import sys
import time

DEFAULT_CHUNK = 30.0
MODEL = 'mlx-community/Qwen3-ASR-1.7B-bf16'
MAX_TOKENS = 220           # see docstring -- do not raise without a reason
REP_PENALTY = 1.15
REP_CONTEXT = 64


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}'


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'csv=p=0', path], capture_output=True, text=True)
    return float(out.stdout.strip())


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    audio, out_srt, lang = sys.argv[1], sys.argv[2], sys.argv[3]
    win = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_CHUNK

    ck = os.path.join(os.path.dirname(out_srt) or '.', 'qwen3_ck')
    os.makedirs(ck, exist_ok=True)

    from mlx_audio.stt.utils import load_model
    dur = duration(audio)
    n = int(dur // win) + 1
    print(f'{n} windows of {win:g}s', flush=True)
    model = load_model(MODEL)

    t0 = time.time()
    for i in range(n):
        chk = f'{ck}/{i:04d}.txt'
        if os.path.exists(chk):
            continue
        clip = f'{ck}/_clip.wav'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', str(i * win),
                        '-t', str(win), '-i', audio, clip], check=True)
        try:
            txt = (model.generate(clip, language=lang,
                                  max_tokens=MAX_TOKENS,
                                  repetition_penalty=REP_PENALTY,
                                  repetition_context_size=REP_CONTEXT).text or '').strip()
        except Exception as exc:                       # one bad window must not kill the run
            txt = ''
            print(f'  window {i} failed: {exc}', flush=True)
        open(chk, 'w', encoding='utf-8').write(txt)
        if i % 20 == 0:
            el = max(time.time() - t0, 1)
            print(f'  {i + 1}/{n} windows, {(i + 1) * win / el:.0f}x realtime', flush=True)

    cues = []
    for i in range(n):
        chk = f'{ck}/{i:04d}.txt'
        if not os.path.exists(chk):
            continue
        txt = open(chk, encoding='utf-8').read().strip()
        if txt:
            cues.append((i * win, min((i + 1) * win, dur), txt))

    with open(out_srt, 'w', encoding='utf-8') as f:
        for k, (s, e, t) in enumerate(cues, 1):
            f.write(f'{k}\n{ts(s)} --> {ts(e)}\n{t}\n\n')
    print(f'wrote {out_srt}: {len(cues)} windows with speech of {n}')


if __name__ == '__main__':
    main()
