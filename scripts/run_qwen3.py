#!/usr/bin/env python3
"""Third transcript from Qwen3-ASR-1.7B via MLX -- independent of BOTH Whisper
and Canary.

    run_qwen3.py <audio.wav> <out.srt> <lang> [target-seconds]

Qwen3-ASR is a separate lineage again (Qwen omni-style encoder + LLM decoder),
so its errors correlate with neither Whisper's nor Canary's. Three witnesses
that disagree independently is what makes "all three agree -> stop thinking
about this cue" a valid shortcut.

Three things this script exists to get right:

1. **No segmentation of its own.** `generate` returns ONE segment per call with
   `start=0.0, end=<clip length>` -- there are no internal timestamps, exactly
   like Canary. So the audio is split into windows here and each window's
   timecode is reconstructed from the actual cut points, not from its index.
   Cue boundaries are still window boundaries, NOT speech boundaries: this file
   is a *wording* source, never a timing source.

2. **Windows are silence-aligned and short (~`target-seconds`, default 8s),
   not fixed 30s blocks** -- same reasoning as `run_canary.py`: a 30s blob of
   prose with one timecode makes it easy to miss that it covers several
   unrelated exchanges, which is exactly what makes a decoder-loop artifact
   elsewhere in the pipeline hard to catch by eye. Costs roughly 4x the model
   calls of one call per 30s.

3. **`max_tokens` defaults to 8192 and that is a live footgun.** A window this
   short holds well under 100 tokens of speech; when the decoder loops it will
   happily generate all 8192, and one window then takes MINUTES instead of ~2 s.
   Measured: a single window burned 5 minutes before being capped. `max_tokens=220`
   plus a repetition penalty removed every loop from an 81-minute film (0 looping
   windows, vs 3 for Canary on the same audio, back when both used 30s windows).

Resume-safe: the cut plan is written to <out-dir>/qwen3_ck/cuts.txt once and
reused, and each window's text is checkpointed to <out-dir>/qwen3_ck/NNNN.txt.
"""
import os
import re
import subprocess
import sys
import time

DEFAULT_TARGET = 8.0
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


def silences(path, noise='-40dB', mind='0.4'):
    r = subprocess.run(
        ['ffmpeg', '-hide_banner', '-i', path, '-af',
         f'silencedetect=noise={noise}:d={mind}', '-f', 'null', '-'],
        capture_output=True, text=True)
    return [float(m) for m in re.findall(r'silence_start:\s*([0-9.]+)', r.stderr)]


def plan_cuts(dur, sil, target):
    """Same planner as chunk_audio.py / run_canary.py."""
    cuts = [0.0]
    while cuts[-1] + target < dur - target / 5:
        want = cuts[-1] + target
        window = target / 6
        cand = [s for s in sil if abs(s - want) < window and s > cuts[-1] + target / 2]
        cuts.append(min(cand, key=lambda s: abs(s - want)) if cand else want)
    cuts.append(dur)
    return cuts


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    audio, out_srt, lang = sys.argv[1], sys.argv[2], sys.argv[3]
    target = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_TARGET

    ck = os.path.join(os.path.dirname(out_srt) or '.', 'qwen3_ck')
    os.makedirs(ck, exist_ok=True)

    cuts_path = os.path.join(ck, 'cuts.txt')
    if os.path.exists(cuts_path):
        cuts = [float(x) for x in open(cuts_path) if x.strip()]
        print(f'reusing {len(cuts) - 1} existing windows from {cuts_path}')
    else:
        dur = duration(audio)
        cuts = plan_cuts(dur, silences(audio), target)
        with open(cuts_path, 'w') as f:
            f.write('\n'.join(f'{c:.3f}' for c in cuts))
    n = len(cuts) - 1
    print(f'{n} windows, target {target:g}s, silence-aligned', flush=True)

    from mlx_audio.stt.utils import load_model
    model = load_model(MODEL)

    t0 = time.time()
    computed = 0
    computed_audio_s = 0.0
    for i in range(n):
        chk = f'{ck}/{i:04d}.txt'
        if os.path.exists(chk):
            continue
        computed += 1
        computed_audio_s += cuts[i + 1] - cuts[i]
        clip = f'{ck}/_clip.wav'
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', str(cuts[i]),
                        '-t', str(cuts[i + 1] - cuts[i]), '-i', audio, clip], check=True)
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
            print(f'  {i + 1}/{n} windows, {computed_audio_s / el:.0f}x realtime', flush=True)

    cues = []
    for i in range(n):
        chk = f'{ck}/{i:04d}.txt'
        if not os.path.exists(chk):
            continue
        txt = open(chk, encoding='utf-8').read().strip()
        if txt:
            cues.append((cuts[i], cuts[i + 1], txt))

    with open(out_srt, 'w', encoding='utf-8') as f:
        for k, (s, e, t) in enumerate(cues, 1):
            f.write(f'{k}\n{ts(s)} --> {ts(e)}\n{t}\n\n')
    print(f'wrote {out_srt}: {len(cues)} windows with speech of {n}')
    if computed:
        elapsed = time.time() - t0
        print(f'{elapsed:.0f}s for {computed_audio_s:.0f}s of new audio '
              f'= {computed_audio_s / max(elapsed, 1e-6):.0f}x realtime')
    else:
        print('all windows were already checkpointed; nothing recomputed')


if __name__ == '__main__':
    main()
