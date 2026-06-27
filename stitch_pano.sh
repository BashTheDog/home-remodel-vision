#!/usr/bin/env bash
# Vertical panorama stitcher
# Usage: stitch_pano.sh <upper_image> <lower_image> <output_name>
# Example: stitch_pano.sh data/pano/upper.jpg data/pano/lower.jpg living_room
#
# Output lands in /project/outputs/pano/<output_name>.jpg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <upper_image> <lower_image> <output_name>"
  echo "  upper_image  — path to the upper photo (absolute or relative)"
  echo "  lower_image  — path to the lower photo (absolute or relative)"
  echo "  output_name  — basename for the output (no extension)"
  exit 1
fi

UPPER="$1"
LOWER="$2"
NAME="$3"
OUTPUT="/project/outputs/pano/${NAME}.jpg"

# Resolve relative paths from the project root
if [[ "$UPPER" != /* ]]; then UPPER="${SCRIPT_DIR}/${UPPER}"; fi
if [[ "$LOWER" != /* ]]; then LOWER="${SCRIPT_DIR}/${LOWER}"; fi

echo "=== Panorama Stitch ==="
echo "  Upper  : $UPPER"
echo "  Lower  : $LOWER"
echo "  Output : $OUTPUT"
echo ""

python3 "${SCRIPT_DIR}/stitch_pano.py" "$UPPER" "$LOWER" "$OUTPUT"

echo ""
echo "=== Done ==="
echo "  Result : $OUTPUT"
echo ""
CONTAINER_IP=$(python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
    s.close()
except Exception:
    print('127.0.0.1')
")

echo "To view in browser:"
echo "  python3 ${SCRIPT_DIR}/tools/pano_viewer/serve.py $OUTPUT"
echo "  Tunnel : ssh -L 8081:${CONTAINER_IP}:8081 <spark-hostname>"
echo "  URL    : http://localhost:8081/"
