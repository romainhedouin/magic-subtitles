#!/usr/bin/env python3
"""Apply Pass B findings to a draft SRT, and split off what's left for the user.

    apply_findings.py <draft.srt> <findings.json> <out.srt> <asks.json>

findings.json:
    {
      "findings": [
        {"cue": 12, "verdict": "fix", "action": "text_fix", "fix_text": "..."},
        {"cue": 47, "verdict": "fix", "action": "retime",
         "new_start": "00:12:03,400", "new_end": "00:12:05,900"},
        {"cue": 8, "verdict": "fix", "action": "drop"},
        {"cue": 540, "verdict": "fix", "action": "reconstruct", "replaces_cues": [540, 556],
         "new_start": "00:43:18,048", "new_end": "00:43:26,048", "fix_text": "..."},
        {"cue": 88, "verdict": "ask", "fix_text": "best guess or empty", "justification": "..."}
      ],
      "drop_matching": ["sous-titrage", "abonnez-vous"],
      "drop_after": 6120.0
    }

Only "fix" findings touch the file. "ask" findings are written to asks.json,
sorted by timestamp, for the user-review table -- never guessed into the output.
"dismiss" findings (and any other verdict) are ignored.

Actions:
  text_fix     -- replace the cue's text; timing untouched unless new_start/new_end given.
  retime       -- replace the cue's timing (and text, if fix_text given).
  drop         -- remove the cue entirely.
  reconstruct  -- like text_fix/retime for a single cue, UNLESS "replaces_cues": [lo, hi]
                  is given: every original cue in that inclusive range is dropped and
                  replaced by every finding sharing that same [lo, hi] pair, ordered by
                  new_start. Use this for a hallucination loop that ate several cues.

Top-level blunt tools, applied before per-cue findings:
  drop_matching -- regexes (case-insensitive) matched against cue text; typically
                   subtitling-house boilerplate ("Sous-titrage ST' 501", "Sous-titrage
                   MFP.") or other repeated hallucinations. Confirm the pattern really
                   is a hallucination (grep the file, look for a run repeating on a
                   fixed interval) before adding it here.
  drop_after    -- drop every cue starting at or after this many seconds. For a
                   hallucinated end-credits tail: confirm the real last line of
                   dialogue first (cross-check against the web reference's last cue),
                   don't just guess a cutoff.

Cues are renumbered sequentially on output, so inserts/drops never leave gaps or
duplicate numbers. Prints a summary -- applied, dropped, skipped -- so nothing
silently fails to land. Re-run qa_srt.py afterward: a retime can overlap an
untouched neighbour, which this script does not check for.
"""
import json
import re
import sys


def to_sec(stamp):
    h, m, rest = stamp.strip().split(':')
    s, ms = rest.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def to_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int(round((sec - int(sec)) * 1000))
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def parse_srt(path):
    blocks = re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip())
    cues = {}
    order = []
    for b in blocks:
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        idx = int(ls[0])
        start, end = ls[1].split('-->')
        cues[idx] = {'start': start.strip(), 'end': end.strip(), 'text': '\n'.join(ls[2:])}
        order.append(idx)
    return cues, order


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    src, findings_path, dst, asks_path = sys.argv[1:5]

    cues, order = parse_srt(src)
    cfg = json.load(open(findings_path, encoding='utf-8'))
    findings = cfg.get('findings', [])

    drop_matching = [re.compile(p, re.I) for p in cfg.get('drop_matching', [])]
    drop_after = cfg.get('drop_after')

    fixes = [f for f in findings if f.get('verdict') == 'fix']
    asks = [f for f in findings if f.get('verdict') == 'ask']
    asks.sort(key=lambda f: f.get('timestamp', ''))
    json.dump(asks, open(asks_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # group reconstruct findings that replace a multi-cue span
    groups = {}
    singles = []
    for f in fixes:
        rc = f.get('replaces_cues')
        span = (rc[0], rc[-1]) if rc else None  # a single-cue span is [n] -> (n, n)
        if span:
            groups.setdefault(span, []).append(f)
        else:
            singles.append(f)

    new_text = {}   # idx -> text (None means drop)
    new_time = {}   # idx -> (start, end)
    applied, skipped = [], []

    for f in singles:
        c = f['cue']
        if c not in cues:
            skipped.append((c, 'cue not found'))
            continue
        action = f.get('action', 'text_fix')
        if action == 'drop':
            new_text[c] = None
        elif action == 'retime':
            new_text[c] = f.get('fix_text', cues[c]['text'])
            new_time[c] = (f.get('new_start', cues[c]['start']), f.get('new_end', cues[c]['end']))
        elif action in ('text_fix', 'reconstruct'):
            new_text[c] = f.get('fix_text', cues[c]['text'])
            if 'new_start' in f and 'new_end' in f:
                new_time[c] = (f['new_start'], f['new_end'])
        else:
            skipped.append((c, f'unknown action {action!r}'))
            continue
        applied.append((c, action))

    replace_at = {}  # lo -> list of {'start','end','text'} in new_start order
    for (lo, hi), members in groups.items():
        if lo not in cues:
            skipped.append((lo, f'reconstruct anchor cue {lo} not found'))
            continue
        members.sort(key=lambda f: f.get('new_start', ''))
        replace_at[lo] = [{'start': m['new_start'], 'end': m['new_end'], 'text': m['fix_text']}
                           for m in members]
        for c in range(lo, hi + 1):
            new_text[c] = None  # dropped; lo's slot gets replaced below
        applied.append((f'{lo}-{hi}', f'reconstruct ({len(members)} cues)'))

    final = []
    for idx in order:
        c = cues[idx]
        start, end, text = c['start'], c['end'], c['text']

        if idx in replace_at:
            final.extend(replace_at[idx])
            continue
        if idx in new_text:
            if new_text[idx] is None:
                continue  # dropped
            text = new_text[idx]
        if idx in new_time:
            start, end = new_time[idx]

        if any(p.search(text) for p in drop_matching):
            continue
        if drop_after is not None and to_sec(start) >= drop_after:
            continue

        final.append({'start': start, 'end': end, 'text': text})

    with open(dst, 'w', encoding='utf-8') as fh:
        for i, c in enumerate(final, 1):
            fh.write(f"{i}\n{c['start']} --> {c['end']}\n{c['text']}\n\n")

    print(f'{len(order)} draft cues -> {len(final)} final cues')
    print(f'{len(applied)} findings applied, {len(skipped)} skipped, {len(asks)} sent to asks.json')
    if skipped:
        print('SKIPPED (check these):')
        for c, why in skipped:
            print(f'  cue {c}: {why}')


if __name__ == '__main__':
    main()
