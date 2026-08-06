#!/usr/bin/env python3
"""Find and repair sentences that start mid-cue with no punctuation (Step 9).

    fix_sentence_breaks.py <in.srt> <lang> --report [--glossary "A,B,C"]
    fix_sentence_breaks.py <in.srt> <lang> --out <out.srt> [--glossary ...]
                           [--words words.json] [--gap 0.35] [--songs-as-breaks]

The pipeline creates this defect, the ASR does not. Whisper emits a segment with
sentence-final punctuation; align_words.py keeps only the words and build_srt.py
regroups them by acoustic onset, so a boundary that used to be a full stop
becomes an ordinary inter-word gap:

    Aide-moi à retourner le canot Mais que faisais-tu là-haut ?

**Why a naive regex is useless.** `[a-zà-ÿ] [A-Z]` fires on every proper noun --
`c'est John Smith`, `du général Li` -- and buries the real cases. Measured on the
reference run: 145 naive hits, 68 genuine. Pass the Step 1 glossary via
--glossary so names are excluded, and never quote an unfiltered count.

Repair preference, highest first:

1. **Split the cue** when word timings (--words, from Step 6) show a gap of at
   least --gap seconds at the boundary. Best fix: it corrects reading speed too.
   Adds a cue; every original timestamp is preserved and the new boundary lies
   strictly inside the cue it came from. Asserted before writing.
2. **Insert a line break** when the gap is too small to split but both halves fit.
3. **Insert a full stop** otherwise.

**Songs are the exception.** Sung lyrics capitalise each line and carry no
terminal punctuation, so `Au détour de la rivière Il sera là` is two lyric lines,
not a missing period. Those must get a line break, never a full stop. This script
cannot tell song from dialogue, so with --songs-as-breaks it NEVER inserts a full
stop and only ever splits or breaks; review the report and choose per film.
"""
import argparse
import json
import re
import sys

SPACE_BEFORE = {'fr'}          # languages putting a space before ! ? ; :
CAP = r"[A-ZÀ-ÂÄ-ÏÑ-ÖÙ-Ý]"
LOW = r"[a-zà-ÿ]"


def parse(path):
    cues = []
    for block in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        lines = block.split('\n')
        if len(lines) < 3:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)', lines[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        cues.append({'idx': lines[0],
                     'start': g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     'end': g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                     'text': '\n'.join(' '.join(l.split()) for l in lines[2:])})
    return cues


def ts(sec):
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}'


# Words that cannot end a sentence. Without this the detector "fixes"
# `où est le Capitaine Smith` into `où est le. Capitaine Smith` -- measured: 4 such
# false repairs on the reference run before the filter existed.
NON_FINAL = {
    'le', 'la', 'les', "l'", 'un', 'une', 'des', 'du', 'de', 'au', 'aux', 'ce', 'cet',
    'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses',
    'notre', 'nos', 'votre', 'vos', 'leur', 'leurs', 'et', 'ou', 'mais', 'donc', 'or',
    'ni', 'car', 'que', "qu'", 'qui', 'dont', 'où', 'à', 'en', 'dans', 'sur', 'sous',
    'par', 'pour', 'avec', 'sans', 'chez', 'vers', 'depuis', 'grand', 'grande',
    'petit', 'petite', 'monsieur', 'madame', 'capitaine', 'gouverneur', 'chef',
    'général', 'roi', 'reine', 'père', 'mère', 'the', 'a', 'an', 'of', 'to', 'and',
}


def boundaries(text, gloss):
    """Offsets where a capitalised word starts a new sentence unmarked.

    A newline already marks the boundary (that is remedy 2), so a capital right
    after one is NOT a defect -- otherwise every repaired cue reports as still
    broken and the loop never converges.
    """
    out = []
    # the capital may be an elided form -- C'est, J'ai, L'or, Qu'il -- so allow an
    # apostrophe inside the token; matching only {CAP}{LOW}+ silently misses these
    for m in re.finditer(rf"(?<![.!?:;…])([ \t]+)({CAP}(?:{LOW}|['’])+)", text):
        if m.group(2) in gloss:
            continue
        if re.fullmatch(r'[IVXLC]+(er|re|e|ème)?', m.group(2)):
            continue                       # roman numeral: "le roi Jacques Ier"
        prev = text[:m.start()].rstrip()
        if not prev or not re.search(rf"[{LOW[1:-1]}0-9]$", prev):
            continue                       # after punctuation or a newline: fine
        last = re.search(rf"[{LOW[1:-1]}'’-]+$", prev)
        if last and last.group(0).lower() in NON_FINAL:
            continue                       # determiner/preposition: not a sentence end
        out.append((m.start(), m.group(2)))
    return out


def wrap(text, limit=42):
    text = ' '.join(text.split())
    if len(text) <= limit:
        return text
    words, best = text.split(), None
    for i in range(1, len(words)):
        a, b = ' '.join(words[:i]), ' '.join(words[i:])
        if len(a) > limit + 3 or len(b) > limit + 3:
            continue
        score = abs(len(a) - len(b)) + (-12 if a.rstrip()[-1:] in '.!?:;,' else 0)
        if best is None or score < best[0]:
            best = (score, a + '\n' + b)
    return best[1] if best else text


def load_gaps(path):
    """word-gap lookup: sorted list of (start, end) for every aligned word."""
    if not path:
        return []
    raw = json.load(open(path))
    words = []
    if isinstance(raw, dict):
        for ws in raw.values():
            words.extend(ws)
    else:
        for win in raw:
            words.extend(win.get('words', []))
    return sorted((float(w['start']), float(w['end'])) for w in words if 'start' in w)


def gap_at(gaps, lo, hi):
    """Largest inter-word silence inside [lo, hi]; (size, midpoint) or (0, None).

    Binary-searched: a linear scan here is O(words x cues) and turns a 2-second
    run into minutes on a feature film (6 800 words x 950 cues).
    """
    import bisect
    i = bisect.bisect_left(gaps, (lo, float('-inf')))
    j = bisect.bisect_right(gaps, (hi, float('inf')))
    inside = gaps[i:j]
    best = (0.0, None)
    for (_, e1), (s2, _) in zip(inside, inside[1:]):
        d = s2 - e1
        if d > best[0]:
            best = (d, e1 + d / 2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('lang')
    ap.add_argument('--report', action='store_true',
                    help='list defects and the planned remedy; write nothing')
    ap.add_argument('--out', default=None, help='output .srt (required unless --report)')
    ap.add_argument('--glossary', default='')
    ap.add_argument('--words', default=None)
    ap.add_argument('--gap', type=float, default=0.35)
    ap.add_argument('--songs-as-breaks', action='store_true')
    ap.add_argument('--song-ranges', default='',
                    help='"MM:SS-MM:SS,..." cues inside these never get a full stop')
    a = ap.parse_args()
    if not a.report and not a.out:
        ap.error('give --out <file> or --report')

    def mmss(x):
        p = [float(v) for v in x.split(':')]
        return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]
    songs = []
    for r in filter(None, (r.strip() for r in a.song_ranges.split(','))):
        lo, hi = r.split('-')
        songs.append((mmss(lo), mmss(hi)))

    def is_song(c):
        return a.songs_as_breaks or any(lo <= c['start'] <= hi for lo, hi in songs)

    gloss = {w.strip() for w in a.glossary.split(',') if w.strip()}
    cues = parse(a.src)
    gaps = load_gaps(a.words)
    report = a.report

    hits = [(c, boundaries(c['text'], gloss)) for c in cues]
    hits = [(c, b) for c, b in hits if b]

    if report:
        print(f'{len(hits)} genuine unpunctuated sentence starts '
              f'(glossary: {len(gloss)} names excluded)\n')
        for c, b in hits:
            off, word = b[0]
            g, _ = gap_at(gaps, c['start'], c['end']) if gaps else (0.0, None)
            plan = 'SPLIT' if g >= a.gap else 'break/punct'
            print(f"  {ts(c['start'])[:8]}  [{plan}{f' {g:.2f}s' if gaps else ''}]  "
                  f"…{word}…  {c['text'][:88]}")
        return

    out, splits, breaks, puncts = [], 0, 0, 0
    for c in cues:
        b = boundaries(c['text'], gloss)
        if not b:
            out.append(c)
            continue
        # Cut the cue into sentence pieces at EVERY boundary in one go. Repairing
        # them one at a time does not terminate: flattening the text to re-wrap
        # erases a newline inserted earlier, which resurrects that boundary.
        # An EXISTING newline is already a boundary marker, so it is a piece
        # separator too. Ignoring it is what makes this oscillate: flattening the
        # text to repair boundary A erases the newline at B, which then reappears
        # as a defect on the next pass, and so on forever.
        pieces = []
        for line in c['text'].split('\n'):
            rest = line
            while True:
                bs = boundaries(rest, gloss)
                if not bs:
                    break
                o = bs[0][0]
                pieces.append(' '.join(rest[:o].split()))
                rest = rest[o:].lstrip()
            pieces.append(' '.join(rest.split()))
        pieces = [p for p in pieces if p]

        song = is_song(c)
        # A cue holds at most 2 lines. Two pieces share a cue only if BOTH fit on
        # one line; a piece longer than that needs both lines to itself, so it
        # gets its own cue and is wrapped at write time.
        groups, cur = [], []
        for p in pieces:
            if len(p) > 45:
                if cur:
                    groups.append(cur); cur = []
                groups.append([p])
                continue
            cur.append(p)
            if len(cur) == 2:
                groups.append(cur); cur = []
        if cur:
            groups.append(cur)
        if len(groups) == 1 and all(len(p) <= 45 for p in groups[0]):
            text = ('\n' if song else '. ').join(groups[0])
            if song:
                breaks += len(groups[0]) - 1
            else:
                puncts += len(groups[0]) - 1
            out.append({**c, 'text': text})
            continue

        # Too much for one cue: split by character share of the cue's own span.
        # A sentence boundary is known to be here, so this is safe without a gap.
        total = sum(len(' '.join(g)) for g in groups) or 1
        t, dur = c['start'], c['end'] - c['start']
        for k, g in enumerate(groups):
            share = len(' '.join(g)) / total
            end = c['end'] if k == len(groups) - 1 else min(t + dur * share, c['end'])
            if end <= t:
                end = min(t + 0.3, c['end'])
            out.append({'idx': c['idx'], 'start': t, 'end': end,
                        'text': ('\n' if song else '. ').join(g)})
            t = end
        splits += len(groups) - 1

    # never invent or lose an original boundary
    orig = {(round(c['start'], 3), round(c['end'], 3)) for c in cues}
    kept = {round(c['start'], 3) for c in out} | {round(c['end'], 3) for c in out}
    for s, e in orig:
        assert s in kept and e in kept, f'lost original timestamp {ts(s)}'

    with open(a.out, 'w', encoding='utf-8') as f:
        for i, c in enumerate(out, 1):
            body = c['text'] if '\n' in c['text'] else wrap(c['text'])
            f.write(f"{i}\n{ts(c['start'])} --> {ts(c['end'])}\n{body}\n\n")
    print(f'{len(cues)} -> {len(out)} cues | {splits} split, {breaks} line-broken, '
          f'{puncts} punctuated')
    if puncts and not a.songs_as_breaks:
        print('  check the punctuated ones: a sung line wants a break, not a full stop')


if __name__ == '__main__':
    main()
