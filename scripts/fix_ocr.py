#!/usr/bin/env python3
"""Repair systematic OCR errors in a subtitle file.

    fix_ocr.py <raw.srt> <out.srt> <lang>

Handles the confusions tesseract reliably makes on subtitle bitmaps. Anything
ambiguous is left alone -- run verify_ocr.py afterwards and check the remaining
suspects against their source images before "fixing" them by hand.
"""
import re
import sys

# Languages that put a space before these marks.
SPACE_BEFORE = {'fr'}

# û/î are frequently read as ü/ï; ï+i doubles up. These are shape confusions,
# not spelling, so they are safe as blanket substitutions.
DIACRITIC = [
    (r'ï[ií]', 'î'),
]


def fix_line(ln, lang):
    orig = ln

    # '!' is rendered as 'I!', '!I', or a trailing 'I'
    ln = ln.replace('I!', '!')
    ln = re.sub(r'!I(?![A-Za-zÀ-ÿ])', '!', ln)
    ln = re.sub(r'(?<=[a-zàâäéèêëîïôöùûüçA-Z0-9.?])\s+I$', ' !', ln)

    # stray brace/pipe artifacts
    ln = re.sub(r'\{u\b', 'tu', ln)
    ln = ln.replace('{', '').replace('|', 'l')

    for pat, rep in DIACRITIC:
        ln = re.sub(pat, rep, ln)

    # quote normalisation
    ln = (ln.replace('’', "'").replace('‘', "'")
            .replace('“', '"').replace('”', '"'))

    # spacing
    ln = re.sub(r'\s+([,.])', r'\1', ln)
    if lang in SPACE_BEFORE:
        ln = re.sub(r'\s*([!?;:])', r' \1', ln)
    else:
        ln = re.sub(r'\s+([!?;:])', r'\1', ln)
    ln = re.sub(r'\s{2,}', ' ', ln).strip()

    return ln, ln != orig


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, lang = sys.argv[1], sys.argv[2], sys.argv[3]

    blocks = re.split(r'\n\s*\n', open(src, encoding='utf-8').read().strip())
    out, changed = [], 0

    for b in blocks:
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        timing, body = ls[1], ls[2:]
        fixed = []
        for ln in body:
            new, did = fix_line(ln, lang)
            changed += did
            if new:
                fixed.append(new)
        if fixed:
            out.append((timing, '\n'.join(fixed)))

    with open(dst, 'w', encoding='utf-8') as f:
        for i, (timing, text) in enumerate(out, 1):
            f.write(f'{i}\n{timing}\n{text}\n\n')

    print(f'{len(out)} cues written, {changed} lines corrected')


if __name__ == '__main__':
    main()
