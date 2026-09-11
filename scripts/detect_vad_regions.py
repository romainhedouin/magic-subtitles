#!/usr/bin/env python3
"""Detect real speech regions across the whole film with Silero VAD.

    detect_vad_regions.py <audio.wav> <out.json>

Whisper's own segment-level timestamps are not reliable enough to bound word
alignment (see align_words.py's --vad flag): a wrong segment window lets
wav2vec2 stretch or mis-anchor a word across a real pause. Silero VAD gives an
independent, audio-only signal for where speech actually is, at fine
resolution, across the whole film in one pass -- not per 120s chunk, so its
regions aren't subject to whatever cut points chunk_audio.py picked.

Output: a flat JSON list of {"start": <abs_s>, "end": <abs_s>} speech regions,
covering the whole input file.
"""
import json
import sys
import wave

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps


def read_wav_16k_mono(path):
    """Read a 16-bit PCM mono 16kHz WAV directly, skipping torchaudio/sox --
    silero_vad's own read_audio needs a sox backend that isn't installed here,
    and this pipeline's audio.wav is already exactly this format (Step 3)."""
    with wave.open(path, 'rb') as w:
        assert w.getframerate() == 16000, f'expected 16kHz, got {w.getframerate()}'
        assert w.getnchannels() == 1, f'expected mono, got {w.getnchannels()} channels'
        assert w.getsampwidth() == 2, f'expected 16-bit PCM, got {8 * w.getsampwidth()}-bit'
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    audio_path, out_path = sys.argv[1:3]

    model = load_silero_vad()
    wav = read_wav_16k_mono(audio_path)
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=16000, return_seconds=True,
        min_silence_duration_ms=300, speech_pad_ms=30,
    )
    regions = [{'start': round(t['start'], 3), 'end': round(t['end'], 3)} for t in timestamps]

    json.dump(regions, open(out_path, 'w'), indent=1)
    print(f'{len(regions)} speech regions -> {out_path}')
    if regions:
        total_speech = sum(r['end'] - r['start'] for r in regions)
        print(f'total speech {total_speech:.0f}s over {regions[-1]["end"]:.0f}s of audio')


if __name__ == '__main__':
    main()
