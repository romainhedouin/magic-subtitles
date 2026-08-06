#!/usr/bin/env python3
"""Durable ledger for Pass B, so no observation is lost between reading and reporting.

    findings.py init    <ledger.jsonl> <draft.srt> --flagged <cues.txt>
    findings.py add     <ledger.jsonl> --cue N --verdict fix|ask|dismiss
                        [--was "..."] [--now "..."] [--note "..."]
    findings.py check   <ledger.jsonl>
    findings.py table   <ledger.jsonl> <draft.srt> [--ref reference.srt]
    findings.py fixes   <ledger.jsonl> > fixes.json

**The problem this exists for.** On the reference run a cue was correctly
identified as wrong during the line-by-line pass -- the reference disagreed, the
note "probably 'leur en causera'" was made -- and then never reached the
user-review table. It was found only much later, by accident. Nothing was
careless: the finding simply lived in working memory between reading a batch and
writing the table at the end, and evaporated.

So the deliverables are GENERATED from this ledger rather than recalled. Write a
line the moment you judge a cue; write the table and the fixes file from what is
written down.

Every flagged cue must be dispositioned exactly once:

    fix      you are correcting it now, with justification (--was/--now)
    ask      only the user's ear can settle it -> goes in the review table
    dismiss  examined and judged fine -> --note must say why

`check` enforces the accounting identity

    flagged == fix + ask + dismiss

and exits non-zero if it does not balance, naming the cues that are missing a
disposition. A pass that does not balance is not finished. This is the same
discipline the skill already applies to regex fixes ("verify every pattern
landed"), applied to the reasoning stage -- which is where this one was lost.
"""
import argparse
import json
import os
import re
import sys

VERDICTS = ('fix', 'ask', 'dismiss')


def parse_srt(path):
    cues = {}
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', ls[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        cues[int(ls[0])] = {'t': f'{g[0]:02d}:{g[1]:02d}:{g[2]:02d}',
                            'sec': g[0]*3600 + g[1]*60 + g[2] + g[3]/1000,
                            'text': ' '.join(ls[2:])}
    return cues


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def meta_path(ledger):
    return ledger + '.flagged'


def cmd_init(a):
    nums = []
    for line in open(a.flagged, encoding='utf-8'):
        m = re.match(r'\s*(\d+)', line)
        if m:
            nums.append(int(m.group(1)))
    json.dump(sorted(set(nums)), open(meta_path(a.ledger), 'w'))
    open(a.ledger, 'a', encoding='utf-8').close()
    print(f'{len(set(nums))} cues flagged for disposition -> {meta_path(a.ledger)}')


def cmd_add(a):
    if a.verdict not in VERDICTS:
        sys.exit(f'verdict must be one of {VERDICTS}')
    if a.verdict == 'dismiss' and not a.note:
        sys.exit('dismiss requires --note explaining why the cue is fine')
    if a.verdict == 'fix' and not (a.was and a.now):
        sys.exit('fix requires --was and --now')
    rec = {'cue': a.cue, 'verdict': a.verdict}
    for k in ('was', 'now', 'note'):
        if getattr(a, k):
            rec[k] = getattr(a, k)
    with open(a.ledger, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'cue {a.cue}: {a.verdict}')


def cmd_check(a):
    recs = load(a.ledger)
    flagged = set(json.load(open(meta_path(a.ledger)))) if os.path.exists(meta_path(a.ledger)) else set()
    seen = {}
    dupes = []
    for r in recs:
        if r['cue'] in seen:
            dupes.append(r['cue'])
        seen[r['cue']] = r['verdict']
    counts = {v: sum(1 for x in seen.values() if x == v) for v in VERDICTS}
    missing = sorted(flagged - set(seen))
    extra = sorted(set(seen) - flagged)

    print(f'flagged   : {len(flagged)}')
    for v in VERDICTS:
        print(f'  {v:<8}: {counts[v]}')
    print(f'  total   : {len(seen)}')
    ok = True
    if missing:
        ok = False
        print(f'\nNOT DISPOSITIONED ({len(missing)}): {missing[:40]}'
              + (' …' if len(missing) > 40 else ''))
    if dupes:
        ok = False
        print(f'\nDISPOSITIONED TWICE ({len(dupes)}): {sorted(set(dupes))[:40]}')
    if extra:
        print(f'\nnote: {len(extra)} dispositioned cues were not in the flagged set '
              f'(fine -- found by reading, not by the detector)')
    print('\nBALANCED' if ok else '\nNOT BALANCED -- the pass is not finished')
    sys.exit(0 if ok else 1)


def cmd_table(a):
    recs = {r['cue']: r for r in load(a.ledger) if r['verdict'] == 'ask'}
    cues = parse_srt(a.draft)
    ref = parse_srt(a.ref) if a.ref else {}
    ref_list = sorted(((v['sec'], v['text']) for v in ref.values()), key=lambda x: x[0])
    print('| Time | File says | Reference | Note |')
    print('|---|---|---|---|')
    for n in sorted(recs, key=lambda n: cues.get(n, {}).get('sec', 0)):
        c = cues.get(n)
        if not c:
            continue
        near = [t for s, t in ref_list if abs(s - c['sec']) <= 4]
        print(f"| {c['t']} | `{c['text'].replace(chr(10), ' ')}` | "
              f"{' / '.join(near)[:70] or '*(none)*'} | {recs[n].get('note', '')} |")


def cmd_fixes(a):
    out = []
    for r in load(a.ledger):
        if r['verdict'] == 'fix':
            out.append([re.escape(r['was']).replace(r'\ ', r'\s+'), r['now']])
    json.dump({'replacements': out, 'drop_matching': []},
              sys.stdout, ensure_ascii=False, indent=1)
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    i = sub.add_parser('init'); i.add_argument('ledger'); i.add_argument('draft')
    i.add_argument('--flagged', required=True); i.set_defaults(fn=cmd_init)

    d = sub.add_parser('add'); d.add_argument('ledger')
    d.add_argument('--cue', type=int, required=True)
    d.add_argument('--verdict', required=True)
    d.add_argument('--was'); d.add_argument('--now'); d.add_argument('--note')
    d.set_defaults(fn=cmd_add)

    c = sub.add_parser('check'); c.add_argument('ledger'); c.set_defaults(fn=cmd_check)

    t = sub.add_parser('table'); t.add_argument('ledger'); t.add_argument('draft')
    t.add_argument('--ref'); t.set_defaults(fn=cmd_table)

    f = sub.add_parser('fixes'); f.add_argument('ledger'); f.set_defaults(fn=cmd_fixes)

    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
