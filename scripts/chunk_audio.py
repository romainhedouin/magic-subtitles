#!/usr/bin/env python3
"""Split audio into silence-aligned chunks so WhisperX never runs out of memory.

    chunk_audio.py <audio.wav> <outdir> [target-seconds]

Boundaries are snapped to detected silences so no cut lands mid-word.
Writes <outdir>/cNN.wav and <outdir>/cuts.txt (chunk start offsets, seconds).
"""
import os
import re
import subprocess
import sys


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


def plan(dur, sil, target):
    cuts = [0.0]
    while cuts[-1] + target < dur - target / 5:
        want = cuts[-1] + target
        window = target / 6
        cand = [s for s in sil if abs(s - want) < window and s > cuts[-1] + target / 2]
        cuts.append(min(cand, key=lambda s: abs(s - want)) if cand else want)
    cuts.append(dur)
    return cuts


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, out_dir = sys.argv[1], sys.argv[2]
    target = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    os.makedirs(out_dir, exist_ok=True)

    dur = duration(src)
    sil = silences(src)
    cuts = plan(dur, sil, target)
    print(f'{dur:.0f}s audio, {len(sil)} silences -> {len(cuts) - 1} chunks')

    for i in range(len(cuts) - 1):
        start, length = cuts[i], cuts[i + 1] - cuts[i]
        dst = os.path.join(out_dir, f'c{i:02d}.wav')
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f'  c{i:02d} exists, skipping')
            continue
        subprocess.run(
            ['ffmpeg', '-y', '-v', 'error', '-ss', f'{start:.3f}',
             '-t', f'{length:.3f}', '-i', src, '-c', 'copy', dst], check=True)
        print(f'  c{i:02d}: {start:8.1f} -> {cuts[i + 1]:8.1f} ({length:.0f}s)')

    with open(os.path.join(out_dir, 'cuts.txt'), 'w') as f:
        f.write('\n'.join(f'{c:.3f}' for c in cuts))


if __name__ == '__main__':
    main()
