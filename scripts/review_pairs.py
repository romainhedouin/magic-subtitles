#!/usr/bin/env python3
"""Side-by-side view of the ASR draft against every other source, for line-by-line review.

    review_pairs.py <draft.srt> <from-cue> <to-cue> LABEL=file.srt [LABEL=file.srt ...]

    # the three-source read this pipeline is built around:
    review_pairs.py draft.srt 1 150 CAN=canary.srt REF=reference.srt WEB=web.srt

    # add FRW when Step 5c ran (French audio only):
    review_pairs.py draft.srt 1 150 CAN=canary.srt FRW=french.srt REF=reference.srt

    # legacy two-argument form, still accepted:
    review_pairs.py draft.srt reference.srt 1 150

Prints each draft cue with whatever each source says at the same timecode.
Work through the whole film in batches of 100-150 cues -- this is the pass that
catches errors made of valid words, which a dictionary check cannot see
("les uns" for "les Huns", "Meuf" for "Boeuf", "la passe" for "la face").

Each source answers a different question, and they are not interchangeable:

  ASR  Whisper draft -- the timing authority, and the wording being judged.
  CAN  Canary transcript of the same audio. An independent listener, so when it
       agrees with ASR the line is almost certainly right, and when it differs
       the audio is genuinely ambiguous. Its cues are fixed windows, not speech
       boundaries -- never take timing from it.
  FRW  large-v3-french transcript (Step 5c, French only). A Whisper fine-tune, so
       NOT an independent listener -- agreement with ASR proves little. Consult it
       where ASR has "..." or nothing over obvious dialogue: it recovers speech the
       others drop. Prone to long repetition loops; never take timing from it.
  REF  Subtitle track from the movie file, in the dub language. A different
       translation of the same scene: a strong error *detector*, a weak error
       *corrector*.
  WEB  Target-language subtitle found online. Virtually always translated from
       the English original rather than transcribed from the dub, so it can never
       settle wording -- but it is the best source for proper-noun spellings and
       for what the scene actually means. Not independent of REF.
"""
import os
import re
import sys

WINDOW = 1.2   # seconds of slack: sources segment differently


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


def parse_sources(argv):
    """Returns (draft_path, lo, hi, [(label, cues)], [specs]). Accepts the legacy
    <draft> <reference> <lo> <hi> form so older invocations keep working."""
    if len(argv) == 5 and not re.fullmatch(r'\d+', argv[2]):
        specs = [f'REF={argv[2]}']
        return argv[1], int(argv[3]), int(argv[4]), [('REF', parse(argv[2]))], specs
    if len(argv) < 5:
        sys.exit(__doc__)
    sources, specs = [], []
    for spec in argv[4:]:
        if '=' not in spec:
            sys.exit(f'source must be LABEL=path, got: {spec}\n\n{__doc__}')
        label, path = spec.split('=', 1)
        if not os.path.exists(path):
            sys.exit(f'no such file: {path}')
        sources.append((label.upper()[:3], parse(path)))
        specs.append(spec)
    return argv[1], int(argv[2]), int(argv[3]), sources, specs


def main():
    draft_path, lo, hi, sources, specs = parse_sources(sys.argv)
    draft = parse(draft_path)
    width = max([3] + [len(lb) for lb, _ in sources])

    shown = 0
    for d in draft:
        if not (lo <= d['n'] <= hi):
            continue
        mm, ss = int(d['s'] // 60), int(d['s'] % 60)
        print(f"{d['n']:>4} {mm:02d}:{ss:02d} {'ASR':>{width}}| {d['t']}")
        for label, cues in sources:
            hits = [c['t'].replace('\n', ' ') for c in cues
                    if c['e'] > d['s'] - WINDOW and c['s'] < d['e'] + WINDOW]
            body = ' / '.join(hits)[:200] if hits else '(nothing at this timecode)'
            print(f"{'':>9} {label:>{width}}| {body}")
        print()
        shown += 1

    print(f'--- {shown} cues shown ({lo}-{hi} of {len(draft)}) ---')
    if hi < len(draft):
        nxt = min(hi + (hi - lo + 1), len(draft))
        print(f"next batch: review_pairs.py {draft_path} {hi + 1} {nxt} {' '.join(specs)}")


if __name__ == '__main__':
    main()
