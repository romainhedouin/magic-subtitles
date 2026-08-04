#!/usr/bin/env bash
# Transcribe each audio chunk with WhisperX, checkpointing per chunk.
#
#     run_chunks.sh <chunkdir> <outdir> <model> <lang> [device] [compute]
#
# Safe to re-run: chunks that already produced a non-empty SRT are skipped, so
# a kill or crash costs only the chunk that was in flight.
set -uo pipefail

CHUNKS="${1:?chunk dir}"
OUT="${2:?output dir}"
MODEL="${3:-large-v3}"
LANG="${4:-fr}"
DEVICE="${5:-cpu}"
COMPUTE="${6:-int8}"

mkdir -p "$OUT"
LOG="$OUT/whisperx.log"
fail=0

for f in "$CHUNKS"/c*.wav; do
  b="$(basename "$f" .wav)"
  if [ -s "$OUT/$b.srt" ]; then
    echo "== $b already done, skipping"
    continue
  fi
  echo "== $b starting $(date +%H:%M:%S)"

  OMP_NUM_THREADS=8 whisperx "$f" \
    --model "$MODEL" --language "$LANG" \
    --device "$DEVICE" --compute_type "$COMPUTE" \
    --batch_size 4 --threads 8 \
    --output_format srt --output_dir "$OUT" >>"$LOG" 2>&1
  rc=$?

  # Check the artefact, not the exit code: an OOM kill (137) can be masked by
  # a wrapper reporting success.
  if [ -s "$OUT/$b.srt" ]; then
    echo "== $b OK $(date +%H:%M:%S) cues=$(grep -cE '^[0-9]+$' "$OUT/$b.srt")"
  else
    echo "== $b FAILED rc=$rc $(date +%H:%M:%S) -- see $LOG"
    [ "$rc" -eq 137 ] && echo "   exit 137 = OOM killed; use smaller chunks or --batch_size 1"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "ALLDONE"
else
  echo "ALLDONE (with failures) -- re-run this command to retry only what is missing"
fi
exit "$fail"
