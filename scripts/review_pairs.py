#!/usr/bin/env python3
"""Side-by-side view of the ASR draft against the reference, for line-by-line review.

    review_pairs.py <draft.srt> <reference.srt> <from-cue> <to-cue>

Prints each draft cue with whatever the reference says at the same timecode.
Work through the whole film in batches of 100-150 cues -- this is the pass that
catches errors made of valid words, which a dictionary check cannot see
("les uns" for "les Huns", "Meuf" for "Boeuf", "la passe" for "la face").
"""
import re
import sys


def parse(path):
    cues = []
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)', ls[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        cues.append({'n': int(ls[0]),
                     's': g[0]*3600 + g[1]*60 + g[2] + g[3]/1000,
                     'e': g[4]*3600 + g[5]*60 + g[6] + g[7]/1000,
                     't': ' '.join(ls[2:])})
    return cues


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    draft, reference = parse(sys.argv[1]), parse(sys.argv[2])
    lo, hi = int(sys.argv[3]), int(sys.argv[4])

    shown = 0
    for d in draft:
        if not (lo <= d['n'] <= hi):
            continue
        # widen slightly: the two tracks segment differently, so a draft cue
        # often spans more than one reference cue
        hits = [r['t'].replace('\n', ' ') for r in reference
                if r['e'] > d['s'] - 1.2 and r['s'] < d['e'] + 1.2]
        mm, ss = int(d['s'] // 60), int(d['s'] % 60)
        print(f"{d['n']:>4} {mm:02d}:{ss:02d} ASR| {d['t']}")
        if hits:
            print(f"            REF| {' / '.join(hits)[:135]}")
        else:
            print("            REF| (nothing at this timecode)")
        print()
        shown += 1

    print(f'--- {shown} cues shown ({lo}-{hi} of {len(draft)}) ---')
    if hi < len(draft):
        print(f'next batch: review_pairs.py {sys.argv[1]} {sys.argv[2]} '
              f'{hi + 1} {min(hi + (hi - lo + 1), len(draft))}')


if __name__ == '__main__':
    main()
