#!/usr/bin/env python3
"""Show every source's content across a flagged span, so a decoder-loop word
never gets replaced on vibes alone.

    check_swallowed_spans.py <suspects.json> <out.srt> SRC=path.srt [SRC2=path2.srt ...]

`suspects.json` comes from clamp_durations.py: words whose pre-clamp duration
was implausibly long (Whisper's own segment length tops out at 30s, so that's
the size of the damage a stuck decoder can do). A short interjection does not
need 27 seconds to say -- the question this script exists to force is not
"what word was this" but "how much of this span is actually unaccounted for."

For each suspect it prints:
  - every source's text overlapping the span (CAN/QWN/REF/WEB -- pass whatever
    you have; each is a plain SRT path)
  - how many cues `out.srt` (your current draft/final) has inside that same
    span, and their total text length

A suspect where independent sources show several sentences and your own
output shows one short cue (or none) is not resolved by picking a plausible
word for the original artifact -- that's the signature of swallowed dialogue,
and the fix is the same "targeted re-transcription + cross-source" treatment
as any other coverage gap, applied to this span specifically rather than left
to a later, easy-to-miss QA sweep.
"""
import re
import sys


def parse_time(t):
    h, m, rest = t.split(':')
    s, ms = rest.replace('.', ',').split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_srt(path):
    cues = []
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        m = re.match(r'([\d:,.]+)\s*-->\s*([\d:,.]+)', ls[1])
        if not m:
            continue
        cues.append((parse_time(m.group(1)), parse_time(m.group(2)),
                     ' '.join(ls[2:]).replace('\n', ' ')))
    cues.sort()
    return cues


def overlap_text(cues, t0, t1, pad=10.0):
    """10s default: wide enough to catch a full neighbouring Canary/Qwen3
    window (they run ~8s by default) even when the flagged span sits near a
    window boundary. A narrower pad reliably produces false "nothing here"
    reads against short-windowed sources -- the content is one window over,
    not missing."""
    return [(s, e, txt) for s, e, txt in cues if e >= t0 - pad and s <= t1 + pad]


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    import json
    suspects = json.load(open(sys.argv[1]))
    out_srt = sys.argv[2]
    sources = {}
    for arg in sys.argv[3:]:
        label, path = arg.split('=', 1)
        sources[label] = load_srt(path)
    out_cues = load_srt(out_srt)

    if not suspects:
        print('no flagged spans -- nothing to check')
        return

    for s in suspects:
        t0, t1 = s['abs_start'], s['abs_end']
        print(f"\n=== span {t0:.1f}-{t1:.1f}s ({s['raw_duration_s']}s raw, "
              f"chunk {s['chunk']}, word {s['word'][:30]!r}) ===")

        own = overlap_text(out_cues, t0, t1)
        own_chars = sum(len(txt) for _, _, txt in own)
        print(f"  YOUR OUTPUT: {len(own)} cue(s), {own_chars} chars")
        for cs, ce, txt in own:
            print(f"    [{cs:.1f}-{ce:.1f}] {txt[:100]}")

        max_other_chars = 0
        for label, cues in sources.items():
            ov = overlap_text(cues, t0, t1)
            chars = sum(len(txt) for _, _, txt in ov)
            max_other_chars = max(max_other_chars, chars)
            print(f"  {label}: {len(ov)} cue(s), {chars} chars")
            for cs, ce, txt in ov:
                print(f"    [{cs:.1f}-{ce:.1f}] {txt[:200]}")

        if max_other_chars > own_chars * 2 and max_other_chars > 40:
            print(f"  >>> LIKELY SWALLOWED DIALOGUE: at least one independent "
                  f"source has {max_other_chars} chars here against your "
                  f"{own_chars} -- do not accept a single-word fix for this "
                  f"span without reconciling the difference.")


if __name__ == '__main__':
    main()
