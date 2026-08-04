#!/usr/bin/env python3
"""Apply verified text corrections to a subtitle file without touching timing.

    apply_fixes.py <in.srt> <out.srt> <fixes.json>

fixes.json:
    {
      "replacements": [["\\bMouchou\\b", "Mushu"], ["\\bgonc\\b", "gong"]],
      "drop_cues_between": [4880, 4980],
      "drop_if_foreign": "en",
      "drop_threshold": 0.34
    }

Only "replacements" is required. Every entry should have been confirmed against
the reference transcript first -- see adjudicate.py.

The script asserts that no timestamp changed, and reports every dropped cue so
the deletions can be audited.
"""
import json
import re
import sys

FOREIGN = {
    # Unambiguously English only -- adding "a"/"on"/"or"/"in" would delete
    # French dialogue such as "On a vu" (scores 67% "English").
    'en': set("all believe can decide don't down eyes feel find got heart heavens "
              "know make need see take guy crashing that's you've your just then "
              "whenever truth world light true when you're open lies way let "
              "through too turn what where look far the of it is".split()),
}


def to_sec(stamp):
    h, m, rest = stamp.split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, fixes_path = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = json.load(open(fixes_path, encoding='utf-8'))

    replacements = cfg.get('replacements', [])
    lo, hi = cfg.get('drop_cues_between', [None, None])
    foreign = FOREIGN.get(cfg.get('drop_if_foreign'), set())
    threshold = cfg.get('drop_threshold', 0.34)

    blocks = re.split(r'\n\s*\n', open(src, encoding='utf-8').read().strip())
    kept, dropped, nfix = [], [], 0

    for b in blocks:
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        timing, text = ls[1], '\n'.join(ls[2:])
        start = to_sec(timing.split('-->')[0].strip())

        if foreign and lo is not None and lo <= start <= hi:
            words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ']+", text)]
            if words and sum(1 for w in words if w in foreign) / len(words) > threshold:
                dropped.append((timing, text))
                continue

        new = text
        for pat, rep in replacements:
            new = re.sub(pat, rep, new)
        if new != text:
            nfix += 1
        kept.append((timing, new))          # timing string reused verbatim

    with open(dst, 'w', encoding='utf-8') as f:
        for i, (timing, text) in enumerate(kept, 1):
            f.write(f'{i}\n{timing}\n{text}\n\n')

    # verify nothing about the timing changed
    orig_timings = [b.split('\n')[1] for b in blocks if len(b.split('\n')) >= 3]
    new_timings = [t for t, _ in kept]
    invented = set(new_timings) - set(orig_timings)

    print(f'{len(kept)} cues | {nfix} corrected | {len(dropped)} dropped')
    print(f'timing lines invented or altered: {len(invented)}'
          f'{"  <-- BUG, investigate" if invented else "  (none)"}')
    if dropped:
        print('\ndropped cues (verify each really is unwanted):')
        for timing, text in dropped:
            print(f'  {timing.split("-->")[0].strip()} | {text[:60]!r}')


if __name__ == '__main__':
    main()
