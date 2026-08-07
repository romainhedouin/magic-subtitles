#!/usr/bin/env python3
"""Clamp implausible word durations, and flag any that hid real dialogue.

    clamp_durations.py <words.json> <cuts.txt> <out_words.json> <out_suspects.json> [flag_threshold_s]

The aligner can only place words inside the segment window Whisper gave it, so
a decoder loop or a held vocalization can inflate one word to absorb an entire
segment -- Whisper's own segment length caps out at 30s, so that's the
characteristic size of the damage. Clamping the duration (so the cue doesn't
sprawl and tear neighbouring sentences apart) is necessary but not sufficient:
a word that was 27s long before clamping was standing in for roughly 27
seconds of *something*, and there is no guarantee that something was silence.

This script does both steps together on purpose. Clamping alone silently
discards the evidence that something needs checking -- by the time you notice
a cue reads "Non !" there is nothing left in the data to tell you it used to
span half a minute. Any word whose ORIGINAL (pre-clamp) duration exceeds
`flag_threshold_s` (default 6s -- generous headroom over a real held scream,
which rarely runs past 2-3s) is written to `out_suspects.json` with its
absolute time span, chunk offset included, so `check_swallowed_spans.py` can
be pointed at exactly that window before any replacement word is accepted.
"""
import json
import sys

DEFAULT_FLAG_THRESHOLD = 6.0


def cap_for(word):
    return min(4.5, max(1.2, 0.12 * len(word.strip())))   # generous for sung/held notes


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    words_path, cuts_path, out_words_path, out_suspects_path = sys.argv[1:5]
    flag_threshold = float(sys.argv[5]) if len(sys.argv) == 6 else DEFAULT_FLAG_THRESHOLD

    cuts = [float(x) for x in open(cuts_path) if x.strip()]
    by_chunk = json.load(open(words_path))

    suspects = []
    touched = 0
    total = 0
    for i, key in enumerate(sorted(by_chunk)):
        offset = cuts[i] if i < len(cuts) else 0.0
        for w in by_chunk[key]:
            total += 1
            raw_dur = w['end'] - w['start']
            cap = cap_for(w['word'])
            if raw_dur > flag_threshold:
                suspects.append({
                    'chunk': key,
                    'word': w['word'],
                    'abs_start': round(w['start'] + offset, 3),
                    'abs_end': round(w['end'] + offset, 3),
                    'raw_duration_s': round(raw_dur, 2),
                })
            if raw_dur > cap:
                w['end'] = w['start'] + cap
                touched += 1

    json.dump(by_chunk, open(out_words_path, 'w'), ensure_ascii=False)
    json.dump(suspects, open(out_suspects_path, 'w'), ensure_ascii=False, indent=1)

    print(f'clamped {touched}/{total} words ({100 * touched / max(total, 1):.2f}%)')
    print(f'flagged {len(suspects)} word(s) with raw duration > {flag_threshold:g}s '
          f'-> {out_suspects_path}')
    if suspects:
        print('DO NOT treat these as resolved until check_swallowed_spans.py has run on them:')
        for s in suspects:
            print(f"  {s['abs_start']:.1f}-{s['abs_end']:.1f}s "
                  f"({s['raw_duration_s']}s raw) chunk {s['chunk']}: {s['word'][:40]!r}")


if __name__ == '__main__':
    main()
