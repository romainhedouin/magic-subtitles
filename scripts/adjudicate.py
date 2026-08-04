#!/usr/bin/env python3
"""Show what the reference transcript says where a suspect word appears.

    adjudicate.py <subs.srt> <reference.srt> <lang> WORD [WORD ...]

The reference is usually a different translation from the dub, so it will not
match word for word. Use it to work out what a garbled word must have been --
never to overwrite the dub's wording wholesale.

If the reference is too divergent to tell, leave the word alone and say so.
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
        cues.append({'n': ls[0],
                     's': g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     'e': g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                     't': ' '.join(ls[2:])})
    return cues


def ts(x):
    return f'{int(x // 60):02d}:{int(x % 60):02d}'


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    subs, ref, _lang = sys.argv[1], sys.argv[2], sys.argv[3]
    suspects = sys.argv[4:]

    dub = parse(subs)
    reference = parse(ref)
    pat = re.compile(r'\b(' + '|'.join(re.escape(s) for s in suspects) + r')\b',
                     re.IGNORECASE)

    shown = set()
    for d in dub:
        m = pat.search(d['t'])
        if not m:
            continue
        key = m.group(1).lower()
        if key in shown:
            continue
        shown.add(key)
        window = ' / '.join(
            r['t'].replace('\n', ' ') for r in reference
            if r['e'] > d['s'] - 1.5 and r['s'] < d['e'] + 1.5)
        print(f"[{m.group(1)}]  cue {d['n']} @ {ts(d['s'])}")
        print(f"   ASR : {d['t']}")
        print(f"   REF : {window[:160] or '(no reference cue here)'}")
        print()

    missing = [s for s in suspects if s.lower() not in shown]
    if missing:
        print('not found in subtitles:', ', '.join(missing))


if __name__ == '__main__':
    main()
