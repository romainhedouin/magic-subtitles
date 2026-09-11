#!/usr/bin/env python3
"""Re-anchor words that wav2vec2 placed across a real pause, using Silero VAD.

    vad_reanchor.py <words_tagged.json> <srtdir> <cuts.txt> <vad_regions.json> <out_words.json>

Root cause (confirmed against real data, not assumed): whisperx's word-level
forced alignment front-loads a segment's recognized words toward whichever
edge of its given time window is unconstrained, whenever that window contains
a real stretch of silence next to the actual speech. This shows up as two
distinct, independently-detectable signatures inside one alignment window:

  A. LEADING silence absorbed into the first word of the window. The word's
     own raw duration balloons (already flagged by clamp_durations.py's
     suspects mechanism) because there's nothing before it to bound its
     start, so the aligner backs it up to the window's own start. Example:
     a one-word sentence-opener sitting in an ~8s window gets assigned a
     6.9s duration and anchored to the window's own start, even though it's
     actually spoken about 7s in, right before the rest of the sentence --
     the word's END lands right (anchored by real acoustic content
     immediately following it); only its START is wrong.

  B. TRAILING silence left unclaimed after the last word of the window. No
     single word's duration looks wrong (each is a normal, short duration),
     but the whole recognized cluster sits compressed near the window's own
     start, leaving a multi-second gap between the last word's end and the
     window's own declared end that no word accounts for. Example: a short
     two-word phrase in a ~4.4s window gets compressed into the window's
     first 1.5s, even though it's actually spoken starting about 3.5s in.

Both are corrected by consulting Silero VAD's independent, audio-only speech
regions (detect_vad_regions.py) -- never by re-running whisperx.align with a
modified window (tried once, caused net regressions -- merging/clipping
segments before alignment broke the aligner's own internal backtracking).
This only nudges already-computed word timings that these two checkable
signatures flag; segments showing neither are left byte-for-byte unchanged.

Words are grouped by their own (seg_start, seg_end) tag -- the specific
whisperx-internal output window that produced them -- rather than by
re-matching against the original SRT's segment list by position. That
positional match was tried first and is unsound: whisperx.align() can split
one input segment into several internally (sentence-final punctuation), so a
word's positional index does not reliably identify which window produced it
(measured: one chunk went from 32 input segments to 37 output ones). Grouping
by the tag whisperx itself reports sidesteps that class of bug entirely.

Detector B needs a second, different reference on top of that grouping.
Checked directly against real output: whisperx's own reported seg_start/
seg_end is NOT the search window it was given -- it's a tight retrofit around
just the words it placed (measured directly: seg_end == the last word's own
end, to the millisecond). So a window's *own* tag can never show
unclaimed trailing time; the real search bound CTC was given is the original
Whisper segment's declared start/end from the SRT. Detector B therefore
matches each output window back to its original SRT segment **by time
overlap** (never by position -- that's the bug that broke v2's first attempt)
and only flags the last output window inside a given original segment
(earlier ones in a split are already correctly bounded by the next one).
"""
import bisect
import json
import re
import sys

RATIO_THRESHOLD = 3.0     # detector A: raw duration vs cap_for() ratio
MIN_LEAD_GAP = 1.5        # detector A: minimum raw duration to even consider (s)
MIN_TRAIL_GAP = 1.5       # detector B: minimum unclaimed trailing gap to flag (s)
MIN_SHIFT = 0.3           # detector B: ignore shifts smaller than this (not worth it, likely noise)


def cap_for(word):
    return min(4.5, max(1.2, 0.12 * len(word.strip())))   # same formula as clamp_durations.py


def chunk_vad(regions, lo, hi):
    """VAD regions overlapping [lo, hi) absolute time, converted chunk-relative."""
    out = [{'start': max(0.0, r['start'] - lo), 'end': min(hi - lo, r['end'] - lo)}
           for r in regions if r['end'] > lo and r['start'] < hi]
    out.sort(key=lambda r: r['start'])
    return out


def region_containing(regions, t, tol=0.15):
    for r in regions:
        if r['start'] - tol <= t <= r['end'] + tol:
            return r
    return None


def parse_srt(path):
    """Original Whisper segments -- the true search bound CTC was given, not
    reconstructable from whisperx's own (retrofit-tight) output windows."""
    segs = []
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)', ls[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        text = ' '.join(ls[2:]).strip()
        if not text or not re.search(r'[A-Za-zÀ-ÿ0-9]', text):
            continue
        segs.append({'start': g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     'end':   g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000})
    return segs


def match_original(orig_segs, t, tol=0.2):
    """The original SRT segment containing time t (by overlap, not position).
    Exact containment first -- segments routinely sit back-to-back with a
    ~0s gap, so applying `tol` before checking exact containment can match
    the WRONG (preceding) segment whenever t falls inside that tolerance
    band. Only fall back to the nearest tolerance-padded match if no segment
    exactly contains t."""
    for s in orig_segs:
        if s['start'] <= t <= s['end']:
            return s
    best, best_d = None, tol
    for s in orig_segs:
        d = max(s['start'] - t, t - s['end'], 0.0)
        if d <= best_d:
            best, best_d = s, d
    return best


def main():
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    words_path, srt_dir, cuts_path, vad_path, out_path = sys.argv[1:6]

    by_chunk = json.load(open(words_path))
    cuts = [float(x) for x in open(cuts_path) if x.strip()]
    vad_regions = json.load(open(vad_path))

    a_flagged = a_fixed = b_flagged = b_fixed = b_clipped = 0
    out = {}

    for i, key in enumerate(sorted(by_chunk)):
        lo, hi = cuts[i], cuts[i + 1] if i + 1 < len(cuts) else cuts[i]
        vad = chunk_vad(vad_regions, lo, hi)
        starts = [r['start'] for r in vad]
        orig_segs = parse_srt(f'{srt_dir}/{key}.srt')

        words = [dict(w) for w in by_chunk[key]]   # copy, don't mutate input
        by_seg = {}
        for w in words:
            by_seg.setdefault((w['seg_start'], w['seg_end']), []).append(w)
        # process windows in time order so "last window of this original segment" is well-defined
        seg_keys = sorted(by_seg)

        # group output windows by which original SRT segment they overlap (by time, not position)
        orig_id = {}
        for win_start, win_end in seg_keys:
            m = match_original(orig_segs, win_start)
            orig_id[(win_start, win_end)] = id(m) if m else None
        last_window_of = {}   # original-segment id -> last (seg_start,seg_end) key seen
        for sk in seg_keys:
            oid = orig_id[sk]
            if oid is not None:
                last_window_of[oid] = sk

        for si, (seg_start, seg_end) in enumerate(seg_keys):
            seg_words = by_seg[(seg_start, seg_end)]
            seg_words.sort(key=lambda w: w['start'])

            # --- Detector A: leading silence absorbed into the first word ---
            first = seg_words[0]
            raw_dur = first['end'] - first['start']
            if raw_dur >= MIN_LEAD_GAP and raw_dur > RATIO_THRESHOLD * cap_for(first['word']):
                a_flagged += 1
                r = region_containing(vad, first['end'])
                if r and r['start'] > first['start'] + 0.05:
                    print(f'{key} win[{seg_start:.2f}-{seg_end:.2f}] A: {first["word"]!r} '
                          f'{first["start"]+lo:.2f}-{first["end"]+lo:.2f} '
                          f'(raw {raw_dur:.1f}s) -> start {r["start"]+lo:.2f} '
                          f'(gain +{r["start"]-first["start"]:.1f}s)')
                    first['start'] = r['start']
                    a_fixed += 1

            # --- Detector B: trailing dead air inside the ORIGINAL segment's
            # true search window, only checked on the last output window that
            # original segment produced ---
            oid = orig_id[(seg_start, seg_end)]
            orig = match_original(orig_segs, seg_start)
            if oid is None or last_window_of.get(oid) != (seg_start, seg_end) or orig is None:
                continue
            last = seg_words[-1]
            trailing_gap = orig['end'] - last['end']
            if trailing_gap >= MIN_TRAIL_GAP:
                b_flagged += 1
                cluster_start = seg_words[0]['start']
                idx = bisect.bisect_right(starts, last['end'])
                target = None
                for r in vad[idx:]:
                    if r['start'] < orig['end'] + 0.05:
                        target = r
                        break
                if target:
                    shift = target['start'] - cluster_start
                    if shift >= MIN_SHIFT:
                        clip_at = None
                        if si + 1 < len(seg_keys):
                            nxt_words = by_seg[seg_keys[si + 1]]
                            clip_at = min(w['start'] for w in nxt_words) - 0.02
                        text_preview = ' '.join(w['word'] for w in seg_words)[:40]
                        print(f'{key} win[{seg_start:.2f}-{seg_end:.2f}] B: {text_preview!r} cluster '
                              f'{cluster_start+lo:.2f}-{last["end"]+lo:.2f} -> '
                              f'{cluster_start+shift+lo:.2f}-{last["end"]+shift+lo:.2f} '
                              f'(shift +{shift:.1f}s, orig window end {orig["end"]+lo:.2f})')
                        for w in seg_words:
                            w['start'] += shift
                            w['end'] += shift
                        if clip_at is not None and seg_words[-1]['end'] > clip_at:
                            print(f'  clipped trailing overlap with next window '
                                  f'({seg_words[-1]["end"]+lo:.2f} -> {clip_at+lo:.2f})')
                            seg_words[-1]['end'] = max(clip_at, seg_words[-1]['start'] + 0.05)
                            b_clipped += 1
                        b_fixed += 1

        out[key] = [{'word': w['word'], 'start': w['start'], 'end': w['end']} for w in words]

    json.dump(out, open(out_path, 'w'), ensure_ascii=False)
    print(f'\nwrote {out_path}')
    print(f'detector A (leading silence in first word): {a_flagged} flagged, {a_fixed} corrected')
    print(f'detector B (trailing dead air, front-loaded cluster): {b_flagged} flagged, '
          f'{b_fixed} corrected, {b_clipped} clipped for next-window overlap')


if __name__ == '__main__':
    main()
