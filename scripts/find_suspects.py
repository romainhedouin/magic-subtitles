#!/usr/bin/env python3
"""Surface likely ASR errors by dictionary-checking a subtitle file.

    find_suspects.py <subs.srt> <lang> [--foreign en]

Prints three groups:
  * lowercase words no dictionary recognises  -- usually misheard common words
  * capitalised unknowns                      -- usually proper nouns
  * cues dominated by foreign-language words  -- usually an end-credits song

Nothing is corrected here. Feed the suspects to adjudicate.py to see what the
reference transcript says at the same timecode.
"""
import re
import subprocess
import sys
from collections import Counter

# Unambiguously English tokens only. Never add words that also exist in the
# target language ("a", "on", "or", "in" are all French) or genuine dialogue
# gets flagged and dropped.
FOREIGN = {
    'en': set("all believe can decide don't down eyes feel find got heart heavens "
              "know make need see take guy crashing that's you've your just then "
              "whenever truth world light true when you're open lies way let "
              "through too turn what where look far the of".split()),
}


def parse(path):
    cues = []
    for b in re.split(r'\n\s*\n', open(path, encoding='utf-8').read().strip()):
        ls = b.split('\n')
        if len(ls) < 3:
            continue
        m = re.match(r'(\d+):(\d+):(\d+)[,.](\d+)', ls[1])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        cues.append({'n': ls[0], 'time': ls[1].split('-->')[0].strip(),
                     'sec': g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     'text': ' '.join(ls[2:])})
    return cues


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, lang = sys.argv[1], sys.argv[2]
    foreign = None
    if '--foreign' in sys.argv:
        foreign = sys.argv[sys.argv.index('--foreign') + 1]

    cues = parse(path)
    text = '\n'.join(c['text'] for c in cues)

    # include capitals in the pattern: [a-zà-ÿ]+ silently clips first letters
    # and invents errors like "Habille" -> "abille"
    words = re.findall(r"[A-Za-zÀ-ÿ]+(?:'[A-Za-zÀ-ÿ]+)?", text)
    freq = Counter(words)
    uniq = sorted(set(words))

    proc = subprocess.run(['hunspell', '-d', lang, '-l'],
                          input='\n'.join(uniq), capture_output=True, text=True)
    unknown = set(proc.stdout.split())

    lower = sorted(w for w in unknown if not w[:1].isupper())
    upper = sorted(w for w in unknown if w[:1].isupper())

    print(f'{len(cues)} cues, {len(uniq)} unique words, {len(unknown)} unknown\n')
    print('=== lowercase suspects (likely misheard words) ===')
    print('  ' + '  '.join(f'{w}({freq[w]})' for w in lower) or '  none')
    print('\n=== capitalised unknowns (likely proper nouns) ===')
    print('  ' + '  '.join(f'{w}({freq[w]})' for w in upper) or '  none')

    if foreign and foreign in FOREIGN:
        print(f'\n=== cues dominated by {foreign} ===')
        fset = FOREIGN[foreign]
        for c in cues:
            w = [x.lower() for x in re.findall(r"[A-Za-zÀ-ÿ']+", c['text'])]
            if w and sum(1 for x in w if x in fset) / len(w) > 0.34:
                print(f"  {c['time']} | {c['text'][:60]}")

    print('\nNote: hunspell splits ligatures, so "bœuf" reports as "uf". '
          'Check the word in context before treating it as an error.')


if __name__ == '__main__':
    main()
