#!/usr/bin/env python3
"""List remaining OCR suspects alongside the bitmap they came from.

    verify_ocr.py <ocr.srt> <workdir> <lang>

Prints each unknown word with the PNG it was read from, so the image can be
opened and checked. Do this before "correcting" anything: a missing accent may
be the track's house style rather than an OCR error, and guessing introduces
errors that look authoritative.

Also reports cues where the two OCR passes disagreed (see pgs_to_srt.py).
"""
import json
import os
import re
import subprocess
import sys


def parse(path):
    cues = []
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        cues.append({'n': int(ls[0]), 'text': '\n'.join(ls[2:])})
    return cues


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    srt, work, lang = sys.argv[1], sys.argv[2], sys.argv[3]

    cues = parse(srt)
    index = {}
    idx_path = os.path.join(work, 'index.json')
    if os.path.exists(idx_path):
        index = {i['n']: i['file'] for i in json.load(open(idx_path))}

    words = {}
    for c in cues:
        for w in re.findall(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)?", c['text']):
            words.setdefault(w, []).append(c['n'])

    proc = subprocess.run(['hunspell', '-d', lang, '-l'],
                          input='\n'.join(sorted(words)),
                          capture_output=True, text=True)
    unknown = sorted(set(proc.stdout.split()))

    print(f'{len(unknown)} unknown words\n')
    for w in unknown:
        cue_ns = words[w][:2]
        print(f'*** {w}')
        for n in cue_ns:
            img = index.get(n, '(image not indexed)')
            print(f'    cue {n}: {img}')
    print('\nOpen the image before changing anything. Unaccented capitals are')
    print('often deliberate in subtitle tracks, not an OCR failure.')

    dis_path = os.path.join(work, 'disagree.json')
    if os.path.exists(dis_path):
        dis = json.load(open(dis_path))
        print(f'\ncues where the two OCR passes disagreed: {len(dis)}')
        for n in dis[:20]:
            print(f'  cue {n}: {index.get(int(n), "")}')


if __name__ == '__main__':
    main()
