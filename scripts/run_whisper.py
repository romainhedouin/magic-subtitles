#!/usr/bin/env python3
"""Batched Whisper large-v3 pass over pre-chunked audio, on Metal via MLX.

    run_whisper.py <chunk-dir> <out-dir> <lang>

Loads the model once and loops over every c*.wav in chunk-dir, instead of
invoking the mlx_whisper CLI per chunk. That distinction matters now that the
default chunk target is 120s (Step 5/chunk_audio.py): a film that used to be
~11 chunks at 600s is ~50 at 120s, and the CLI reloads the model fresh on
every invocation -- measured at roughly a minute of pure model-load overhead
each time. This script pays that cost once.

Resume-safe: skips any chunk whose <name>.srt already exists in out-dir.
"""
import os
import sys
import time

import mlx_whisper

MODEL = 'mlx-community/whisper-large-v3-mlx'


def to_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f'{h:02d}:{m:02d}:{int(s):02d},{int((s % 1) * 1000):03d}'


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    chunk_dir, out_dir, lang = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(chunk_dir) if f.endswith('.wav'))
    t0 = time.time()
    for i, fn in enumerate(files):
        name = fn[:-4]
        out_path = os.path.join(out_dir, f'{name}.srt')
        if os.path.exists(out_path):
            print(f'skip {name} (exists)', flush=True)
            continue
        result = mlx_whisper.transcribe(
            os.path.join(chunk_dir, fn),
            path_or_hf_repo=MODEL,
            language=lang,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2,
        )
        with open(out_path, 'w', encoding='utf-8') as fh:
            for n, seg in enumerate(result['segments'], 1):
                fh.write(f"{n}\n{to_ts(seg['start'])} --> {to_ts(seg['end'])}\n"
                          f"{seg['text'].strip()}\n\n")
        print(f'{i + 1}/{len(files)} {name} done, {time.time() - t0:.0f}s elapsed', flush=True)
    print(f'TOTAL {time.time() - t0:.0f}s for {len(files)} chunks')


if __name__ == '__main__':
    main()
