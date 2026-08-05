#!/bin/bash
# Fetch bofenghuang/whisper-large-v3-french and convert it to MLX for Step 5c.
#
#     bash scripts/setup_whisper_fr.sh [dest-dir]
#
# Default dest: ~/.cache/whisper-mlx/whisper-large-v3-french
# Idempotent: re-running with the model already built is a no-op, and a partial
# download resumes rather than restarting.
#
# There is no published MLX build of this model, so we convert it ourselves.
# Three things bite here, and all three are handled below:
#
#   1. HuggingFace's Xet transfer stalls on this repo. Both snapshot_download and
#      HF_HUB_ENABLE_HF_TRANSFER=1 died at ~90 KB and never recovered, twice.
#      Plain `curl -C -` against resolve/main pulls the full 3.2 GB at 1-7 MB/s.
#   2. mlx-whisper's convert.py is not shipped inside the installed package --
#      it only exists in the mlx-examples repo, so we fetch it.
#   3. convert.py writes `model.safetensors`, but mlx_whisper's load_models.py
#      looks for `weights.safetensors` (then `weights.npz`) and dies with a
#      confusing "[load_npz] Input must be a zip file" if the name is wrong.
set -euo pipefail

REPO=bofenghuang/whisper-large-v3-french
BASE=https://huggingface.co/$REPO/resolve/main
DEST=${1:-$HOME/.cache/whisper-mlx/whisper-large-v3-french}
WORK=$DEST.build
SIZE=3219908024      # bytes of model.safetensors; verified before converting

if [ -f "$DEST/weights.safetensors" ] && [ -f "$DEST/config.json" ]; then
  echo "already built: $DEST"; exit 0
fi

mkdir -p "$WORK/hf"
echo "=== fetching $REPO (3.2 GB) ==="
curl -sfL -o "$WORK/hf/config.json" "$BASE/config.json"
# -C - resumes a partial file; the retry flags cover the transient 5xx that HF
# throws under load. Do NOT swap this for huggingface_hub -- see note 1 above.
curl -L -C - --retry 5 --retry-delay 3 -o "$WORK/hf/model.safetensors" "$BASE/model.safetensors"

GOT=$(stat -f %z "$WORK/hf/model.safetensors" 2>/dev/null || stat -c %s "$WORK/hf/model.safetensors")
if [ "$GOT" != "$SIZE" ]; then
  echo "download incomplete: $GOT bytes, expected $SIZE. Re-run to resume." >&2
  exit 1
fi

echo "=== building conversion venv ==="
uv venv --python 3.12 "$WORK/venv" >/dev/null
uv pip install --python "$WORK/venv/bin/python" -q torch mlx mlx-whisper huggingface_hub tqdm numpy
curl -sfL -o "$WORK/convert.py" \
  https://raw.githubusercontent.com/ml-explore/mlx-examples/main/whisper/convert.py

echo "=== converting to MLX float16 ==="
"$WORK/venv/bin/python" "$WORK/convert.py" \
  --torch-name-or-path "$WORK/hf" --mlx-path "$DEST" --dtype float16

# note 3: rename to what the installed mlx_whisper actually looks for
if [ -f "$DEST/model.safetensors" ]; then
  mv "$DEST/model.safetensors" "$DEST/weights.safetensors"
fi

rm -rf "$WORK"
echo "=== done: $DEST ==="
ls -la "$DEST"
