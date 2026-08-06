#!/usr/bin/env python3
"""Draft vs every source for line-by-line review, with word-level disagreement.

    review_pairs.py <draft.srt> <from-cue> <to-cue> LABEL=file.srt [LABEL=...]
                    [--asr CAN,QWN] [--tier both|one|all] [--rank]

    # the read this pipeline is built around:
    review_pairs.py draft.srt 1 150 CAN=canary.srt QWN=qwen3.srt \\
                    REF=reference.srt WEB=web.srt

    # what to look at first, whole film, highest-signal cues only:
    review_pairs.py draft.srt 1 9999 CAN=canary.srt QWN=qwen3.srt REF=ref.srt \\
                    --rank --tier both

    # legacy two-argument form, still accepted:
    review_pairs.py draft.srt reference.srt 1 150

**Why this marks individual words.** The independent ASRs emit fixed 30-second
windows, so reading a two-second cue against a paragraph leaves you eyeballing
for a one-word swap -- and that is exactly the error class this pass exists to
catch. Comparing whole cues by similarity makes it worse: a homophone swap
changes one token in four (~25% divergence) and sits below any sensible
"this cue looks different" threshold, while genuinely garbled cues score high.
The filter then hides the errors that matter and surfaces the ones you would
have spotted anyway.

So there is no similarity threshold. Each cue's words are sequence-aligned
against the overlapping window text of every ASR source named in --asr, and any
word no source supports is marked [[like this]]. Cues are then tiered by how
many *independent* sources dissent:

    both  every ASR source disagrees on >=1 word  -- read closely
    one   exactly one disagrees                   -- read normally
    none  all support every word                  -- skim

Measured on an 81-minute French film against 21 confirmed corrections: the
"both" tier held 17 of them (81% recall) while being 23% of all cues.

That is a RANKING, not a filter. Four known errors sat outside it -- two where
both models garbled the same noisy passage, two where both models agreed with
the wrong reading. Still read every cue; use the tier to decide how hard.

Each source answers a different question, and they are not interchangeable:

  ASR  Whisper draft -- the timing authority, and the wording being judged.
  CAN  Canary-1B-v2. Independent listener (NeMo FastConformer lineage).
  QWN  Qwen3-ASR-1.7B. Independent again, of Whisper *and* of Canary.
       CAN and QWN both emit fixed windows -- never take timing from either.
  REF  Subtitle track from the movie file, in the dub language. A different
       translation of the same scene: a strong error *detector*, a weak error
       *corrector*.
  WEB  Target-language subtitle found online. Virtually always translated from
       the English original rather than transcribed from the dub, so it can never
       settle wording -- but it is the best source for proper-noun spellings and
       for what the scene actually means. Not independent of REF.
"""
import difflib
import os
import re
import sys

WINDOW = 1.2        # seconds of slack: sources segment differently
DEFAULT_ASR = ('CAN', 'QWN')


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


def norm(w):
    return re.sub(r"[^a-zà-ÿ0-9']", '', w.lower())


def unsupported(cue_words, window_text):
    """Indices of cue words the window does not contain, by sequence alignment.

    Alignment rather than set membership: it respects order, so a word repeated
    in the cue but present once in the window is still caught.
    """
    a = [norm(w) for w in cue_words]
    b = [norm(w) for w in window_text.split()]
    b = [w for w in b if w]
    missing = set(i for i, w in enumerate(a) if w)
    for i, _, n in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks():
        for k in range(n):
            missing.discard(i + k)
    return missing


def parse_args(argv):
    if len(argv) == 5 and not re.fullmatch(r'\d+', argv[2]):
        return argv[1], int(argv[3]), int(argv[4]), [('REF', parse(argv[2]))], \
               [f'REF={argv[2]}'], list(DEFAULT_ASR), 'all', False
    flags = {'asr': list(DEFAULT_ASR), 'tier': 'all', 'rank': False}
    args = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == '--asr':
            i += 1; flags['asr'] = [x.strip().upper()[:3] for x in argv[i].split(',') if x.strip()]
        elif a == '--tier':
            i += 1; flags['tier'] = argv[i]
        elif a == '--rank':
            flags['rank'] = True
        else:
            args.append(a)
        i += 1
    if len(args) < 4:
        sys.exit(__doc__)
    sources, specs = [], []
    for spec in args[3:]:
        if '=' not in spec:
            sys.exit(f'source must be LABEL=path, got: {spec}\n\n{__doc__}')
        label, path = spec.split('=', 1)
        if not os.path.exists(path):
            sys.exit(f'no such file: {path}')
        sources.append((label.upper()[:3], parse(path)))
        specs.append(spec)
    return args[0], int(args[1]), int(args[2]), sources, specs, \
        flags['asr'], flags['tier'], flags['rank']


def main():
    draft_path, lo, hi, sources, specs, asr_labels, tier_want, rank = parse_args(sys.argv)
    draft = parse(draft_path)
    width = max([3] + [len(lb) for lb, _ in sources])
    by_label = dict(sources)
    active_asr = [lb for lb in asr_labels if lb in by_label]

    rows = []
    for d in draft:
        if not (lo <= d['n'] <= hi):
            continue
        words = d['t'].replace('\n', ' ').split()
        wins, dissent = {}, {}
        for lb, cues in sources:
            hits = [c['t'].replace('\n', ' ') for c in cues
                    if c['e'] > d['s'] - WINDOW and c['s'] < d['e'] + WINDOW]
            wins[lb] = ' / '.join(hits) if hits else ''
        for lb in active_asr:
            dissent[lb] = unsupported(words, wins[lb]) if wins[lb] else None
        voting = [v for v in dissent.values() if v is not None]
        if len(voting) >= 2:
            both = set.intersection(*voting)
            n_dis = sum(1 for v in voting if v)
        elif len(voting) == 1:
            both = voting[0]
            n_dis = 1 if voting[0] else 0
        else:
            both, n_dis = set(), 0
        tier = 'both' if (len(voting) >= 2 and both) else ('one' if n_dis else 'none')
        rows.append((d, words, wins, dissent, both, tier))

    order = {'both': 0, 'one': 1, 'none': 2}
    shown = rows if tier_want == 'all' else [r for r in rows if r[5] == tier_want]
    if rank:
        shown = sorted(shown, key=lambda r: (order[r[5]], -len(r[4]), r[0]['s']))

    for d, words, wins, dissent, both, tier in shown:
        mm, ss = int(d['s'] // 60), int(d['s'] % 60)
        marked = ' '.join(f'[[{w}]]' if i in both else w for i, w in enumerate(words))
        tag = {'both': '!!', 'one': ' ~', 'none': '  '}[tier]
        print(f"{d['n']:>4} {mm:02d}:{ss:02d} {tag} {'ASR':>{width}}| {marked}")
        for lb, _ in sources:
            body = wins[lb][:200] if wins[lb] else '(nothing at this timecode)'
            mark = ''
            if lb in dissent and dissent[lb]:
                mark = f" <{len(dissent[lb])} word{'s' if len(dissent[lb]) > 1 else ''} unsupported>"
            print(f"{'':>9}    {lb:>{width}}| {body}{mark}")
        print()

    counts = {t: sum(1 for r in rows if r[5] == t) for t in ('both', 'one', 'none')}
    total = len(rows) or 1
    print(f"--- {len(shown)} shown of {len(rows)} cues in range ({lo}-{hi} of {len(draft)}) ---")
    print(f"    tiers: !! both dissent {counts['both']} ({counts['both']/total:.0%})"
          f" | ~ one {counts['one']} | none {counts['none']}")
    if active_asr:
        print(f"    independent ASR sources compared: {', '.join(active_asr)}")
    else:
        print("    WARNING: no ASR source matched --asr; word marking is off")
    if tier_want == 'all' and not rank and hi < len(draft):
        nxt = min(hi + (hi - lo + 1), len(draft))
        print(f"next batch: review_pairs.py {draft_path} {hi + 1} {nxt} {' '.join(specs)}")


if __name__ == '__main__':
    main()
