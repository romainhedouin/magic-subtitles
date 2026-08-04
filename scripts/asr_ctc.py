#!/usr/bin/env python3
"""Independent second-opinion transcript from a wav2vec2 CTC model.

    asr_ctc.py <chunkdir> <out.json> <hf-model-id>

Why CTC and not a second Whisper run: two Whisper runs share weights and
training data, so they reproduce each other's systematic errors -- a second
Whisper pass will happily agree that "les Huns" is "les uns". A wav2vec2 CTC
model has a different architecture (frame-wise classification, no autoregressive
decoder, no internal language model), so its errors are largely uncorrelated.
That is what makes disagreement informative.

Its output is lowercase and unpunctuated, and it is *worse* than Whisper on its
own. Do not use it as a transcript -- use it only to flag where the two
disagree, then adjudicate those spots.

French: jonatasgrosman/wav2vec2-large-xlsr-53-french
English: jonatasgrosman/wav2vec2-large-xlsr-53-english

Optional 4th arg: window length in seconds (default 6). Keep it short -- see
the note in the loop below.
"""
import json
import os
import sys

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_wav(path):
    import wave
    with wave.open(path, 'rb') as w:
        assert w.getframerate() == 16000, 'expected 16 kHz audio'
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    chunk_dir, out_path, model_id = sys.argv[1:4]
    global WINDOW
    WINDOW = int(sys.argv[4]) if len(sys.argv) > 4 else 6

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()

    out = {}
    for fn in sorted(os.listdir(chunk_dir)):
        if not fn.endswith('.wav'):
            continue
        audio = load_wav(os.path.join(chunk_dir, fn))
        words = []
        # Short windows matter: cross_check.py can only localise a disagreement
        # to the window it fell in. With 30 s windows the comparison bag holds
        # ~90 words and real disagreements are lost in it. 6 s keeps it tight.
        step, overlap = WINDOW * 16000, 1 * 16000
        for start in range(0, len(audio), step - overlap):
            seg = audio[start:start + step]
            if len(seg) < 8000:
                break
            inputs = processor(seg, sampling_rate=16000, return_tensors='pt', padding=True)
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            ids = torch.argmax(logits, dim=-1)
            text = processor.batch_decode(ids)[0].strip().lower()
            if text:
                words.append({'t': text, 'start': start / 16000})
        out[fn[:-4]] = words
        print(f'{fn[:-4]}: {sum(len(w["t"].split()) for w in words)} words', flush=True)

    json.dump(out, open(out_path, 'w'))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
