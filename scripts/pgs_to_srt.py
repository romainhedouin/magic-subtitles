#!/usr/bin/env python3
"""Decode a PGS (.sup) bitmap subtitle stream and OCR it into an SRT.

    pgs_to_srt.py <input.sup> <workdir> <tesseract-lang>

Writes <workdir>/sub_NNNNN.png, <workdir>/index.json and <workdir>/raw.srt.
Run the output through fix_ocr.py afterwards.
"""
import json
import os
import re
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image


def rle_decode(raw, width, height):
    """PGS run-length encoding -> flat list of palette indices."""
    out, line = [], []
    i, n = 0, len(raw)
    while i < n:
        b = raw[i]
        i += 1
        if b != 0:
            line.append(b)
            continue
        if i >= n:
            break
        b2 = raw[i]
        i += 1
        if b2 == 0:                                  # end of line
            line.extend([0] * (width - len(line)))
            out.extend(line[:width])
            line = []
            continue
        cnt = b2 & 0x3F
        if b2 & 0x40:
            cnt = (cnt << 8) | raw[i]
            i += 1
        colour = 0
        if b2 & 0x80:
            colour = raw[i]
            i += 1
        line.extend([colour] * cnt)
    if line:
        line.extend([0] * (width - len(line)))
        out.extend(line[:width])
    need = width * height
    return out[:need] + [0] * max(0, need - len(out))


def decode_sup(path):
    """Parse a .sup into a list of display events, each with rendered images."""
    data = open(path, 'rb').read()
    palettes, objects = {}, {}
    events, pending = [], None
    pos = 0

    while pos + 13 <= len(data):
        if data[pos:pos + 2] != b'PG':
            pos += 1
            continue
        pts = struct.unpack('>I', data[pos + 2:pos + 6])[0] / 90000.0
        stype = data[pos + 10]
        slen = struct.unpack('>H', data[pos + 11:pos + 13])[0]
        payload = data[pos + 13:pos + 13 + slen]
        pos += 13 + slen

        if stype == 0x16:                            # presentation composition
            num_comp = payload[10]
            comps, p = [], 11
            for _ in range(num_comp):
                obj_id, _win, flags = struct.unpack('>HBB', payload[p:p + 4])
                p += 8
                if flags & 0x40:
                    p += 8
                comps.append(obj_id)
            if pending is not None:                  # any new composition ends the previous
                pending['end'] = pts
                events.append(pending)
                pending = None
            if comps:
                pending = {'start': pts, 'end': None,
                           'palette_id': payload[9] if len(payload) > 9 else 0}

        elif stype == 0x14:                          # palette definition
            entries = palettes.setdefault(payload[0], {})
            p = 2
            while p + 5 <= len(payload):
                idx, y, cr, cb, a = payload[p:p + 5]
                entries[idx] = (y, cr, cb, a)
                p += 5

        elif stype == 0x15:                          # object definition
            obj_id = struct.unpack('>H', payload[0:2])[0]
            seq = payload[3]
            if seq & 0x80:
                w, h = struct.unpack('>HH', payload[7:11])
                objects[obj_id] = {'w': w, 'h': h, 'data': bytearray(payload[11:])}
            elif obj_id in objects:
                objects[obj_id]['data'].extend(payload[4:])
            if seq & 0x40 and pending is not None:   # last fragment -> render
                o = objects.get(obj_id)
                if not o:
                    continue
                idxs = rle_decode(bytes(o['data']), o['w'], o['h'])
                pal = palettes.get(pending['palette_id'], {})
                img = Image.new('L', (o['w'], o['h']), 255)
                px = img.load()
                for i, ci in enumerate(idxs):
                    y, _cr, _cb, a = pal.get(ci, (0, 128, 128, 0))
                    px[i % o['w'], i // o['w']] = 255 if a < 40 else 255 - y
                pending.setdefault('imgs', []).append(img)

    if pending is not None:
        pending['end'] = pending['start'] + 3.0
        events.append(pending)
    return events


def render(events, out_dir):
    """Stack each event's images, pad and upscale for OCR. Returns an index."""
    index, n = [], 0
    for ev in events:
        imgs = ev.get('imgs') or []
        if not imgs:
            continue
        if len(imgs) == 1:
            img = imgs[0]
        else:
            w = max(i.width for i in imgs)
            h = sum(i.height for i in imgs) + 10 * (len(imgs) - 1)
            img = Image.new('L', (w, h), 255)
            y = 0
            for i in imgs:
                img.paste(i, ((w - i.width) // 2, y))
                y += i.height + 10
        canvas = Image.new('L', (img.width + 40, img.height + 40), 255)
        canvas.paste(img, (20, 20))
        canvas = canvas.resize((canvas.width * 2, canvas.height * 2), Image.LANCZOS)
        n += 1
        fn = os.path.join(out_dir, f'sub_{n:05d}.png')
        canvas.save(fn)
        index.append({'n': n, 'start': ev['start'], 'end': ev['end'], 'file': fn})
    return index


def ocr_all(index, lang, psm='6', oem='1'):
    def one(item):
        r = subprocess.run(
            ['tesseract', item['file'], 'stdout', '-l', lang, '--psm', psm,
             '--oem', oem, '-c', 'preserve_interword_spaces=1'],
            capture_output=True, text=True)
        return item['n'], r.stdout
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(one, index))


def ts(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sup, out_dir, lang = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    events = decode_sup(sup)
    index = render(events, out_dir)
    json.dump(index, open(os.path.join(out_dir, 'index.json'), 'w'))
    print(f'decoded {len(index)} subtitle images')

    results = ocr_all(index, lang)

    # Second pass with different engine settings; disagreements flag low
    # confidence. Cheap, and a useful sanity signal on OCR quality.
    second = ocr_all(index, lang, psm='4', oem='3')
    norm = lambda s: re.sub(r'\s+', ' ', re.sub(r"[^0-9A-Za-zÀ-ÿ' ]", ' ', s)).strip().lower()
    disagree = [n for n in results if norm(results[n]) != norm(second.get(n, ''))]
    print(f'OCR pass-2 disagreement: {len(disagree)}/{len(results)} cues')
    json.dump(disagree, open(os.path.join(out_dir, 'disagree.json'), 'w'))

    out, n = [], 0
    for item in index:
        lines = [re.sub(r'\s+', ' ', l).strip()
                 for l in results.get(item['n'], '').splitlines() if l.strip()]
        if not lines:
            continue
        end = item['end'] if item['end'] and item['end'] > item['start'] else item['start'] + 2.0
        n += 1
        out.append(f'{n}\n{ts(item["start"])} --> {ts(end)}\n' + '\n'.join(lines) + '\n')

    path = os.path.join(out_dir, 'raw.srt')
    open(path, 'w', encoding='utf-8').write('\n'.join(out))
    print(f'wrote {n} cues -> {path}')


if __name__ == '__main__':
    main()
