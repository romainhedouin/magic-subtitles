#!/usr/bin/env python3
"""Alignment-only pass: recover word-level timings WhisperX discards on write.

    align_words.py <chunkdir> <srtdir> <out.json> <lang>

WhisperX aligns to word level internally but writes SRT using whisper's coarse
segment boundaries. This runs the wav2vec2 aligner again over the existing
transcript -- a forward pass only, far cheaper than re-transcribing -- and
keeps the word timings so cues can be re-segmented properly.
"""
import gc
import json
import os
import re
import sys

import whisperx


def parse_srt(path):
    """Segments with text. Junk cues (no alphanumerics) are dropped."""
    segs = []
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)', ls[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        text = ' '.join(ls[2:]).strip()
        if not text or not re.search(r'[A-Za-zÀ-ÿ0-9]', text):
            continue
        segs.append({'start': g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     'end':   g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                     'text':  text})
    return segs


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    chunk_dir, srt_dir, out_path, lang = sys.argv[1:5]

    device = 'cpu'
    model_a, meta = whisperx.load_align_model(language_code=lang, device=device)

    out = {}
    for fn in sorted(os.listdir(chunk_dir)):
        if not fn.endswith('.wav'):
            continue
        b = fn[:-4]
        srt = os.path.join(srt_dir, f'{b}.srt')
        if not os.path.exists(srt):
            print(f'{b}: no srt, skipping', flush=True)
            continue
        segs = parse_srt(srt)
        if not segs:
            print(f'{b}: no usable segments, skipping', flush=True)
            continue

        audio = whisperx.load_audio(os.path.join(chunk_dir, fn))
        res = whisperx.align(segs, model_a, meta, audio, device,
                             return_char_alignments=False)
        # timings come back as numpy floats, which json cannot serialise
        out[b] = [{'word': w['word'], 'start': float(w['start']), 'end': float(w['end'])}
                  for s in res['segments'] for w in s.get('words', [])
                  if 'start' in w and 'end' in w]
        print(f'{b}: {len(segs)} segs -> {len(out[b])} words', flush=True)
        del audio, res
        gc.collect()

    json.dump(out, open(out_path, 'w'))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
