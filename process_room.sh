#!/usr/bin/env bash
# process_room.sh <start_time> <end_time> <room_name> [fps]
#
# Extracts frames from /project/data/IMG_2445.MOV, runs COLMAP SfM,
# and trains a 3D Gaussian Splatting model.
#
# Args:
#   start_time  : ffmpeg time spec, e.g. 00:00:30 or 30
#   end_time    : ffmpeg time spec, e.g. 00:01:10 or 70
#   room_name   : slug used for output dirs, e.g. living_room
#   fps         : frames per second to extract (default: 2)
#
# Outputs:
#   /project/data/frames/<room_name>/         — extracted frames
#   /project/outputs/<room_name>_colmap/      — COLMAP workspace
#   /project/outputs/<room_name>_3dgs/        — trained .ply + checkpoint
#
# Example:
#   bash process_room.sh 00:00:10 00:01:00 kitchen
#   bash process_room.sh 120 180 hallway 3

set -euo pipefail

START="${1:-}"
END="${2:-}"
ROOM="${3:-}"
FPS="${4:-2}"

if [[ -z "$START" || -z "$END" || -z "$ROOM" ]]; then
    echo "Usage: $0 <start_time> <end_time> <room_name> [fps]"
    echo "  e.g. $0 00:00:30 00:01:10 kitchen"
    exit 1
fi

VIDEO="/project/data/IMG_2445.MOV"
FRAMES_DIR="/project/data/frames/${ROOM}"
COLMAP_DIR="/project/outputs/${ROOM}_colmap"
COLMAP_DENSE="${COLMAP_DIR}/dense"
OUTPUT_DIR="/project/outputs/${ROOM}_3dgs"
COLMAP_BIN="/project/tools/envs/colmap/bin/colmap"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Extract frames ──────────────────────────────────────────────────────
log "Step 1/4 — Extracting frames (${FPS} fps, transpose=1, 1280px wide)"
mkdir -p "$FRAMES_DIR"

# transpose=1: 90° clockwise rotation (portrait iPhone video → upright)
# scale=1280:-2: resize width to 1280, height auto (must be even)
ffmpeg -y -ss "$START" -to "$END" -i "$VIDEO" \
    -vf "transpose=1,scale=1280:-2" \
    -q:v 2 \
    -r "$FPS" \
    "${FRAMES_DIR}/frame_%04d.jpg" \
    2>&1 | tail -5

NFRAMES=$(ls "$FRAMES_DIR"/*.jpg 2>/dev/null | wc -l)
log "  Extracted ${NFRAMES} frames → ${FRAMES_DIR}"

if [[ "$NFRAMES" -lt 10 ]]; then
    echo "ERROR: Too few frames ($NFRAMES). Check start/end times."
    exit 1
fi

# ── 2. COLMAP feature extraction + matching + SfM ─────────────────────────
log "Step 2/4 — COLMAP: feature extraction + matching + SfM reconstruction"
mkdir -p "$COLMAP_DENSE/images" "$COLMAP_DENSE/sparse"

# Symlink frames into the COLMAP images directory (avoids copying large files)
for f in "$FRAMES_DIR"/*.jpg; do
    ln -sf "$f" "$COLMAP_DENSE/images/$(basename "$f")"
done

log "  Feature extraction …"
"$COLMAP_BIN" feature_extractor \
    --database_path "${COLMAP_DIR}/database.db" \
    --image_path "${COLMAP_DENSE}/images" \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 1 \
    2>&1 | grep -E "Processed|ERROR|error" || true

log "  Sequential matching …"
"$COLMAP_BIN" sequential_matcher \
    --database_path "${COLMAP_DIR}/database.db" \
    --FeatureMatching.use_gpu 1 \
    2>&1 | grep -E "Matched|ERROR|error" || true

log "  Sparse reconstruction (mapper) …"
"$COLMAP_BIN" mapper \
    --database_path "${COLMAP_DIR}/database.db" \
    --image_path "${COLMAP_DENSE}/images" \
    --output_path "${COLMAP_DENSE}/sparse" \
    2>&1 | grep -E "Registering|Total|ERROR|error" || true

# COLMAP mapper writes to sparse/0/ — move to sparse/ directly if needed
if [[ -d "${COLMAP_DENSE}/sparse/0" ]] && [[ ! -f "${COLMAP_DENSE}/sparse/cameras.bin" ]]; then
    log "  Moving sparse/0/ → sparse/ …"
    mv "${COLMAP_DENSE}/sparse/0/"* "${COLMAP_DENSE}/sparse/"
    rmdir "${COLMAP_DENSE}/sparse/0"
fi

if [[ ! -f "${COLMAP_DENSE}/sparse/cameras.bin" ]]; then
    echo "ERROR: COLMAP mapper failed — cameras.bin not found."
    echo "  Check that enough frames were matched."
    exit 1
fi

NREG=$(python3 -c "
import struct
with open('${COLMAP_DENSE}/sparse/images.bin','rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
print(n)
" 2>/dev/null || echo "?")
log "  Registered ${NREG} images"

# ── 3. 3D Gaussian Splatting training ─────────────────────────────────────
log "Step 3/4 — 3DGS training (this takes ~15–30 min on GB10)"
python3 /project/train_3dgs.py \
    --colmap-dir "$COLMAP_DENSE" \
    --output-dir "$OUTPUT_DIR" \
    --name "$ROOM"

PLY="${OUTPUT_DIR}/${ROOM}.ply"
if [[ ! -f "$PLY" ]]; then
    echo "ERROR: 3DGS training failed — ${PLY} not created."
    exit 1
fi

# ── 4. Done ───────────────────────────────────────────────────────────────
PLY_MB=$(python3 -c "import os; print(f'{os.path.getsize(\"${PLY}\")/1e6:.1f}')")
log "Step 4/4 — Complete!"
echo ""
echo "  PLY splat : ${PLY} (${PLY_MB} MB)"
echo "  Frames    : ${FRAMES_DIR} (${NFRAMES} frames)"
echo "  COLMAP    : ${COLMAP_DENSE}"
echo ""
echo "  To view:  bash /project/view_room.sh ${ROOM}"
