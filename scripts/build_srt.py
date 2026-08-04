#!/usr/bin/env python3
"""Rebuild subtitle cues from word-level timings under real subtitle constraints.

    build_srt.py <words.json> <cuts.txt> <out.srt> <lang>

Whisper's own segments are unusable as cues -- they run to 20+ seconds. This
regroups the aligned words on acoustic boundaries and caps duration, length and
reading speed.
"""
import json
import re
import statistics
import sys

MAX_DUR = 6.0        # seconds per cue
MAX_CHARS = 84       # 2 lines x 42
MAX_LINE = 42
MIN_DUR = 0.9
MAX_GAP = 0.8        # split when silence between words exceeds this
MAX_CPS = 20.0       # characters per second; above this is hard to read

SPACE_BEFORE = {'fr'}


def punctuate(text, lang):
    text = re.sub(r'\s+([,.])', r'\1', text)
    if lang in SPACE_BEFORE:
        text = re.sub(r'\s*([!?;:])', r' \1', text)
    else:
        text = re.sub(r'\s+([!?;:])', r'\1', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def group(words):
    cues, cur = [], []

    def flush():
        if cur:
            cues.append({'s': cur[0]['s'], 'e': cur[-1]['e'],
                         'w': [x['w'] for x in cur]})
            cur.clear()

    for w in words:
        if cur:
            gap = w['s'] - cur[-1]['e']
            dur = w['e'] - cur[0]['s']
            chars = len(' '.join(x['w'] for x in cur)) + 1 + len(w['w'])
            ends_sentence = re.search(r'[.!?]$', cur[-1]['w'])
            if gap > MAX_GAP or dur > MAX_DUR or chars > MAX_CHARS \
                    or (ends_sentence and chars > 40):
                flush()
        cur.append(w)
    flush()
    return cues


def wrap(text):
    if len(text) <= MAX_LINE:
        return text
    words = text.split()
    best, best_cost = None, 1e9
    for i in range(1, len(words)):
        a, b = ' '.join(words[:i]), ' '.join(words[i:])
        if len(a) > MAX_LINE or len(b) > MAX_LINE:
            continue
        cost = abs(len(a) - len(b))
        if re.search(r'[,.!?;:]$', words[i - 1]):
            cost -= 12                          # prefer breaking after punctuation
        if cost < best_cost:
            best, best_cost = (a, b), cost
    if best:
        return '\n'.join(best)
    # no split fits both lines: fall back to the most balanced one
    mid, best_cost = None, 1e9
    for i in range(1, len(words)):
        cost = abs(len(' '.join(words[:i])) - len(' '.join(words[i:])))
        if cost < best_cost:
            mid, best_cost = i, cost
    return '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])]) if mid else text


def ts(x):
    ms = int(round(x * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    words_path, cuts_path, out_path, lang = sys.argv[1:5]

    cuts = [float(x) for x in open(cuts_path) if x.strip()]
    by_chunk = json.load(open(words_path))

    words = []
    for i, key in enumerate(sorted(by_chunk)):
        offset = cuts[i] if i < len(cuts) else 0.0
        for w in by_chunk[key]:
            words.append({'w': w['word'].strip(),
                          's': w['start'] + offset,
                          'e': w['end'] + offset})
    words.sort(key=lambda x: x['s'])

    cues = group(words)

    out = []
    for i, c in enumerate(cues):
        text = punctuate(' '.join(c['w']), lang)
        s, e = c['s'], max(c['e'], c['s'] + MIN_DUR)
        # hold fast cues longer, but only into real silence
        need = len(text) / MAX_CPS
        if e - s < need:
            limit = cues[i + 1]['s'] - 0.08 if i + 1 < len(cues) else c['e'] + 2.5
            e = min(s + need, max(e, limit))
        if i + 1 < len(cues):
            e = min(e, cues[i + 1]['s'] - 0.04)      # never overlap the next cue
        if e <= s:
            e = s + MIN_DUR
        out.append((s, e, wrap(text)))

    with open(out_path, 'w', encoding='utf-8') as f:
        for i, (s, e, text) in enumerate(out, 1):
            f.write(f'{i}\n{ts(s)} --> {ts(e)}\n{text}\n\n')

    durs = [e - s for s, e, _ in out]
    lens = [len(t.replace('\n', ' ')) for _, _, t in out]
    print(f'{len(out)} cues -> {out_path}')
    print(f'median dur {statistics.median(durs):.1f}s  max {max(durs):.1f}s')
    print(f'median chars {statistics.median(lens):.0f}  max {max(lens)}')


if __name__ == '__main__':
    main()
