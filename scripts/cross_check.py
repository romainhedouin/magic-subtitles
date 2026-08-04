#!/usr/bin/env python3
"""Flag words where an independent ASR pass disagrees with the subtitle file.

    cross_check.py <subs.srt> <ctc.json> <cuts.txt> [--min-len 4]

Compares the finished subtitles against a second-opinion transcript produced by
a different model family (see asr_ctc.py). A word present in the subtitles but
absent from the independent pass at the same timecode is a candidate error.

This catches what a reference *translation* cannot: it compares two readings of
the same audio, so it still works inside songs and ad-libs where a translated
reference diverges by design.

Expect false positives. The CTC pass has no language model, so it mangles
proper nouns and drops function words. Treat the output as a shortlist to
listen to, never as corrections to apply.
"""
import difflib
import json
import re
import sys
import unicodedata


# Similarity below this means the independent pass saw nothing like the word.
SIM_THRESHOLD = 0.62

# Function words: CTC drops them constantly and they are never the error anyway.
STOP = set("""alors apres aussi autre avait avant avec beaucoup bien cela cette chez
comme comment dans depuis donc elle elles encore entre etait etaient etre faire fait
ici jamais leur leurs mais meme moins nous parce pareil pendant peut plus point pour
pourquoi quand quel quelle qui quoi sans sera seulement sont sous soyez sur tous tout
toute toutes tres trop vers vous voila votre etais serait seront avons avez ont ete
puis rien peu deja tant celui ceux dont aussi""".split())


def strip_accents(s):
    s = unicodedata.normalize('NFD', s.lower())
    return ''.join(c for c in s if not unicodedata.combining(c))


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r"[^a-z0-9' ]", ' ', strip_accents(s))).strip()


def parse_srt(path):
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
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    subs_path, ctc_path, cuts_path = sys.argv[1:4]
    min_len = 4
    if '--min-len' in sys.argv:
        min_len = int(sys.argv[sys.argv.index('--min-len') + 1])

    cues = parse_srt(subs_path)
    cuts = [float(x) for x in open(cuts_path) if x.strip()]
    by_chunk = json.load(open(ctc_path))

    # Flatten the independent pass onto the film timeline. The window length is
    # inferred from the spacing of consecutive segments -- hardcoding it would
    # silently widen every comparison bag and destroy the signal.
    spans = []
    for i, key in enumerate(sorted(by_chunk)):
        off = cuts[i] if i < len(cuts) else 0.0
        segs = by_chunk[key]
        starts = [s['start'] for s in segs]
        gaps = sorted(b - a for a, b in zip(starts, starts[1:])) or [30.0]
        win = gaps[len(gaps) // 2] + 1.0        # median spacing + the overlap
        for seg in segs:
            spans.append({'s': seg['start'] + off, 'e': seg['start'] + off + win,
                          'w': set(norm(seg['t']).split())})

    flagged = []
    for c in cues:
        near = set()
        for sp in spans:
            if sp['e'] > c['s'] - 2 and sp['s'] < c['e'] + 2:
                near |= sp['w']
        if not near:
            continue
        for w in norm(c['t']).split():
            if len(w) < min_len or w in near or w in STOP:
                continue
            # CTC routinely drops or doubles letters, so an exact miss means
            # little. Only flag when nothing in the window is even close.
            best = max((difflib.SequenceMatcher(None, w, o).ratio() for o in near),
                       default=0.0)
            if best < SIM_THRESHOLD:
                flagged.append((c, w, best))

    print(f'{len(cues)} cues | {len(flagged)} words the independent pass did not confirm\n')
    seen = {}
    for c, w, score in flagged:
        seen.setdefault(c['n'], (c, [], 1.0))
        entry = seen[c['n']]
        entry[1].append(w)
        seen[c['n']] = (c, entry[1], min(entry[2], score))

    # worst agreement first -- those are the likeliest real errors
    for c, words, score in sorted(seen.values(), key=lambda x: x[2]):
        mm, ss = int(c['s'] // 60), int(c['s'] % 60)
        print(f"{mm:02d}:{ss:02d}  cue {c['n']:>4}  [{', '.join(words)}]  (sim {score:.2f})")
        print(f"    {c['t']}")
    print(f'\n{len(seen)} distinct cues flagged. Listen before changing anything.')


if __name__ == '__main__':
    main()
