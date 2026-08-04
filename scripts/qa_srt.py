#!/usr/bin/env python3
"""Structural, readability and coverage QA for a finished subtitle file.

    qa_srt.py <subs.srt> [reference.srt]

With a reference, also reports coverage: moments where the reference has a cue
but the output has none. That is how you find dialogue the ASR missed entirely.
"""
import re
import statistics
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
                     's': g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     'e': g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                     't': '\n'.join(ls[2:])})
    return cues


def ts(x):
    return f'{int(x // 60):02d}:{int(x % 60):02d}'


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cues = parse(sys.argv[1])
    if not cues:
        sys.exit('no cues parsed -- is this an SRT?')

    problems = 0

    print(f'cues: {len(cues)}')
    print(f'span: {ts(cues[0]["s"])} -> {ts(cues[-1]["e"])}')

    overlaps = sum(1 for i in range(len(cues) - 1) if cues[i]['e'] > cues[i + 1]['s'])
    nonmono = sum(1 for i in range(len(cues) - 1) if cues[i]['s'] > cues[i + 1]['s'])
    badlen = sum(1 for c in cues if c['e'] <= c['s'])
    numbering = all(c['n'] == i for i, c in enumerate(cues, 1))
    problems += overlaps + nonmono + badlen + (0 if numbering else 1)

    print(f'overlapping cues:  {overlaps}')
    print(f'non-monotonic:     {nonmono}')
    print(f'zero/negative dur: {badlen}')
    print(f'numbering contiguous: {numbering}')

    durs = [c['e'] - c['s'] for c in cues]
    lens = [len(c['t'].replace('\n', ' ')) for c in cues]
    cps = [l / d for l, d in zip(lens, durs) if d > 0]
    over3 = sum(1 for c in cues if len(c['t'].split('\n')) > 2)
    longline = sum(1 for c in cues for ln in c['t'].split('\n') if len(ln) > 45)
    fast = sum(1 for x in cps if x > 25)

    print(f'\nmedian duration:   {statistics.median(durs):.1f}s (max {max(durs):.1f}s)')
    print(f'median CPS:        {statistics.median(cps):.1f}')
    print(f'cues over 25 CPS:  {fast}  (hard to read)')
    print(f'cues over 2 lines: {over3}')
    print(f'lines over 45 chars: {longline}')

    if len(sys.argv) > 2:
        ref = parse(sys.argv[2])
        missed = [r for r in ref
                  if not any(c['e'] > r['s'] - 0.5 and c['s'] < r['e'] + 0.5 for c in cues)]
        runs = []
        for r in missed:
            if runs and r['s'] - runs[-1][1] < 12:
                runs[-1][1] = r['e']
                runs[-1][2] += 1
            else:
                runs.append([r['s'], r['e'], 1])
        print(f'\nreference cues: {len(ref)}')
        print(f'uncovered:      {len(missed)} ({len(missed) / len(ref):.0%}), '
              f'{sum(r["e"] - r["s"] for r in missed) / 60:.1f} min total')
        big = sorted((r for r in runs if r[2] >= 3), key=lambda r: -(r[1] - r[0]))[:6]
        if big:
            print('largest gaps (investigate: missed dialogue, or a musical passage?)')
            for s, e, n in big:
                print(f'  {ts(s)}-{ts(e)}  {n} cues  ({e - s:.0f}s)')

    print(f'\n{"PASS" if problems == 0 else "ISSUES FOUND"}'
          f' -- structural problems: {problems}')
    return 0 if problems == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
