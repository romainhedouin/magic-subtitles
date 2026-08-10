#!/usr/bin/env python3
"""Find and repair broken sentence boundaries: missing AND spurious (Step 9).

    fix_sentence_breaks.py <in.srt> <lang> --report [--glossary "A,B,C"]
    fix_sentence_breaks.py <in.srt> <lang> --out <out.srt> [--glossary ...]
                           [--words words.json] [--gap 0.35] [--songs-as-breaks]

The pipeline creates the *missing*-punctuation defect, the ASR does not.
Whisper emits a segment with sentence-final punctuation; align_words.py keeps
only the words and build_srt.py regroups them by acoustic onset, so a boundary
that used to be a full stop becomes an ordinary inter-word gap:

    Aide-moi à retourner le canot Mais que faisais-tu là-haut ?

**Why a naive regex is useless for that.** `[a-zà-ÿ] [A-Z]` fires on every
proper noun -- `c'est John Smith`, `du général Li` -- and buries the real
cases. Measured on the reference run: 145 naive hits, 68 genuine. Pass the
Step 1 glossary via --glossary so names are excluded, and never quote an
unfiltered count.

**The opposite defect is at least as common, and this script used to miss it
entirely.** Whisper's OWN punctuation model periodically hallucinates a full
stop at a mid-sentence hesitation or breath pause -- not a pipeline artifact,
a genuine ASR failure mode -- leaving `[.!?]` sitting inside a clause:

    j'ignore. comment          <- should read "j'ignore comment"
    Grand-mère. Feuillage,     <- a single proper noun torn in two

Measured on an 81-minute film already run through every other pass in this
skill (Pass A dictionary check, Pass B contextual review, this script's own
missing-punctuation repair): **16 clean instances** of `[.!?]` directly
followed by a lowercase letter -- a 100%-precision signal, since that
sequence is never grammatical in French or English -- plus several more
where the spurious mark split a multi-word proper noun and the continuation
happened to stay capitalised, which only a glossary check catches. None of
these were caught by Pass A (hunspell checks words, not punctuation
placement), Pass B (subagents were judging *wording*, not stray marks), or
this script's original missing-punctuation detector (which only ever
*inserts*, and has no signal for a mark that is already wrongly *present*).
--report and --out below now check both directions in the same pass; treat a
clean run of this script as covering both, not just the insert side.

**A related, rarer failure: long monologues with NO punctuation at all.**
Fast, unbroken speech (a barked list of orders, e.g.) can make Whisper's
punctuation model produce zero marks across several consecutive cues and
many words. The missing-punctuation detector below is keyed off a *capital
letter* appearing where none is expected -- if Whisper never restored a
single capital in the run either, there is no signal to catch it on. --report
now also flags any run of --long-run-words (default 12) or more consecutive
words with no `[.!?,:;]` anywhere in them; there is no safe auto-fix for
this one (where the clauses actually break is a judgement call), so it is
reported only, for manual punctuation exactly like Step 8's user-review
table.

Repair preference for MISSING punctuation, highest first:

1. **Split the cue** when word timings (--words, from Step 6) show a gap of at
   least --gap seconds at the boundary. Best fix: it corrects reading speed too.
   Adds a cue; every original timestamp is preserved and the new boundary lies
   strictly inside the cue it came from. Asserted before writing.
2. **Insert a line break** when the gap is too small to split but both halves fit.
3. **Insert a full stop** otherwise.

For SPURIOUS punctuation the repair is simpler and has no ambiguity: delete
the mark and rejoin the clause. No timing changes, since it never touches a
cue boundary -- only text within a single existing cue.

**Songs are the exception for MISSING punctuation only.** Sung lyrics
capitalise each line and carry no terminal punctuation, so
`Au détour de la rivière Il sera là` is two lyric lines, not a missing
period. Those must get a line break, never a full stop. This script cannot
tell song from dialogue, so with --songs-as-breaks it NEVER inserts a full
stop and only ever splits or breaks; review the report and choose per film.
Spurious-punctuation removal is unaffected by --songs-as-breaks: a stray mid-
word period is wrong in a song exactly as much as in dialogue.
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


def spurious_boundaries(text, gloss):
    """Offsets of a `[.!?]` mark that is wrongly present mid-clause.

    Two sub-cases, both removed the same way (delete the mark, rejoin):

    1. Followed directly by a lowercase letter -- 100% precision, this
       sequence is never grammatical. Whisper's punctuation model
       hallucinated a full stop at a breath/hesitation pause.
    2. Followed by a capitalised word, but the mark+space+word, once
       removed, reconstitutes a multi-word glossary entry exactly
       (e.g. "Grand-mère. Feuillage" with "Grand-mère Feuillage" in the
       glossary). This is deliberately narrow -- a capital after a real
       mark is usually a genuine new sentence, so only act when the
       glossary itself proves the two pieces are one name.
    """
    out = []
    for m in re.finditer(rf'[.!?](\s+)({LOW}|{CAP}(?:{LOW}|[\'’])+)', text):
        mark_pos = m.start()
        if text[max(0, mark_pos - 1):mark_pos + 1] == '..':
            continue  # ellipsis
        word = m.group(2)
        if word[0].islower():
            out.append(mark_pos)
            continue
        # capitalised continuation: only remove if it completes a glossary name
        before = text[:mark_pos]
        prev_word = re.search(rf"({CAP}(?:{LOW}|[\'’]|-)*)$", before)
        if prev_word and f'{prev_word.group(1)} {word}' in gloss:
            out.append(mark_pos)
    return out


def remove_spurious(text, gloss):
    """Delete every spurious mark found by spurious_boundaries, once each pass
    (positions shift after each removal, so recompute rather than batch-index)."""
    removed = 0
    while True:
        offs = spurious_boundaries(text, gloss)
        if not offs:
            return text, removed
        p = offs[0]
        text = text[:p] + text[p + 1:]
        removed += 1


def unpunctuated_runs(cues, min_words=12):
    """Consecutive cues covering >= min_words with zero [.!?,:;] anywhere.

    Report-only: which clause boundary is correct inside a completely bare
    run is a judgement call (see module docstring), so this never auto-fixes.
    Returns list of (start_cue, end_cue, start_ts, word_count).
    """
    runs = []
    cur_start, cur_words, cur_ts = None, 0, None
    for c in cues:
        has_punct = bool(re.search(r'[.!?,:;]', c['text']))
        words = len(c['text'].split())
        if has_punct or words == 0:
            if cur_start is not None and cur_words >= min_words:
                runs.append((cur_start, c, cur_ts, cur_words))
            cur_start, cur_words, cur_ts = None, 0, None
            continue
        if cur_start is None:
            cur_start, cur_ts = c, c['start']
        cur_words += words
    if cur_start is not None and cur_words >= min_words:
        runs.append((cur_start, cur_start, cur_ts, cur_words))
    return runs


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
    ap.add_argument('--long-run-words', type=int, default=12,
                    help='report (never auto-fix) any run of this many consecutive '
                         'words or more with zero [.!?,:;] anywhere in them')
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
    spurious_hits = [(c, spurious_boundaries(c['text'], gloss)) for c in cues]
    spurious_hits = [(c, o) for c, o in spurious_hits if o]
    bare_runs = unpunctuated_runs(cues, a.long_run_words)

    if report:
        print(f'{len(hits)} genuine unpunctuated sentence starts '
              f'(glossary: {len(gloss)} names excluded)\n')
        for c, b in hits:
            off, word = b[0]
            g, _ = gap_at(gaps, c['start'], c['end']) if gaps else (0.0, None)
            plan = 'SPLIT' if g >= a.gap else 'break/punct'
            print(f"  {ts(c['start'])[:8]}  [{plan}{f' {g:.2f}s' if gaps else ''}]  "
                  f"…{word}…  {c['text'][:88]}")

        print(f"\n{len(spurious_hits)} spurious mid-clause punctuation marks "
              f"(will be deleted, not replaced)\n")
        for c, offs in spurious_hits:
            print(f"  {ts(c['start'])[:8]}  [DELETE {len(offs)}]  {c['text'][:88]}")

        print(f"\n{len(bare_runs)} run(s) of >= {a.long_run_words} words with NO "
              f"punctuation at all (report only -- pick the breaks by hand)\n")
        for start_c, end_c, ts_, words in bare_runs:
            print(f"  {ts(ts_)[:8]}  {words} words, cues {start_c['idx']}-{end_c['idx']}")
        return

    # Spurious marks first: deleting one can only ever create MORE missing-
    # punctuation boundaries (a stray "j'ignore. comment" becomes clean text
    # with no boundary at all here), never fewer -- so this order is safe and
    # the two passes can't fight each other.
    removed_total = 0
    for c in cues:
        c['text'], n = remove_spurious(c['text'], gloss)
        removed_total += n
    if removed_total:
        print(f'removed {removed_total} spurious mid-clause punctuation mark(s)')

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
        #
        # Each piece after the first carries whether the cut BEFORE it was a
        # real detected sentence start (needs a period when rejoined) or just
        # the next original line with no boundary inside it (the newline
        # between them was already a valid boundary marker on its own -- join
        # with a space, not a period). Losing this distinction is exactly what
        # produced bugs like "j'ignore. comment" / "exactement. la même
        # question": two pieces from adjacent original LINES, with no real
        # sentence break between them, got bucketed into one 2-piece output
        # group by the line-fitting logic below and joined with '. ' anyway.
        pieces = []  # list of (text, needs_period_before)
        for line in c['text'].split('\n'):
            line_pieces, rest = [], line
            while True:
                bs = boundaries(rest, gloss)
                if not bs:
                    break
                o = bs[0][0]
                head = ' '.join(rest[:o].split())
                if head:
                    line_pieces.append(head)
                rest = rest[o:].lstrip()
            tail = ' '.join(rest.split())
            if tail:
                line_pieces.append(tail)
            # Only a cut WITHIN this line (between line_pieces[i] and [i+1])
            # is a genuine detected boundary -> needs a period. The join to
            # the previous LINE's last piece is a newline-gap, already a
            # valid boundary marker on its own -> needs_period False, always,
            # regardless of what came before it.
            for i, p in enumerate(line_pieces):
                pieces.append((p, i > 0))

        song = is_song(c)
        # A cue holds at most 2 lines. Two pieces share a cue only if BOTH fit on
        # one line; a piece longer than that needs both lines to itself, so it
        # gets its own cue and is wrapped at write time.
        groups, cur = [], []
        for p in pieces:
            if len(p[0]) > 45:
                if cur:
                    groups.append(cur); cur = []
                groups.append([p])
                continue
            cur.append(p)
            if len(cur) == 2:
                groups.append(cur); cur = []
        if cur:
            groups.append(cur)

        def join_group(g):
            """Join a group's pieces into (text, periods_inserted), respecting
            each piece's own needs_period flag rather than blanket-joining
            with '. ' -- see note above."""
            if song:
                return '\n'.join(p for p, _ in g), 0
            text, n = g[0][0], 0
            for p, needs_period in g[1:]:
                text += ('. ' if needs_period else ' ') + p
                n += needs_period
            return text, n

        if len(groups) == 1 and all(len(p[0]) <= 45 for p in groups[0]):
            text, n = join_group(groups[0])
            if song:
                breaks += len(groups[0]) - 1
            else:
                puncts += n
            out.append({**c, 'text': text})
            continue

        # Too much for one cue: split by character share of the cue's own span.
        # A sentence boundary is known to be here, so this is safe without a gap.
        total = sum(len(' '.join(p for p, _ in g)) for g in groups) or 1
        t, dur = c['start'], c['end'] - c['start']
        for k, g in enumerate(groups):
            share = len(' '.join(p for p, _ in g)) / total
            end = c['end'] if k == len(groups) - 1 else min(t + dur * share, c['end'])
            if end <= t:
                end = min(t + 0.3, c['end'])
            text, n = join_group(g)
            out.append({'idx': c['idx'], 'start': t, 'end': end, 'text': text})
            puncts += n
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

    if bare_runs:
        print(f'\n{len(bare_runs)} unpunctuated run(s) of >= {a.long_run_words} words '
              f'were NOT touched -- pick the sentence breaks by hand:')
        for start_c, end_c, ts_, words in bare_runs:
            print(f"  {ts(ts_)[:8]}  {words} words, cues {start_c['idx']}-{end_c['idx']}")


if __name__ == '__main__':
    main()
