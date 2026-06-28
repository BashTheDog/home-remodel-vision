#!/usr/bin/env bash
# Interactive object removal editor
# Usage: edit.sh <image_path>
# Example: edit.sh outputs/pano/test_stitch.jpg

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <image_path>"
  exit 1
fi

IMAGE="$1"
if [[ "$IMAGE" != /* ]]; then IMAGE="${SCRIPT_DIR}/${IMAGE}"; fi

CONTAINER_IP=$(python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()
")

echo "=== Interactive Editor ==="
echo "  Image  : $IMAGE"
echo "  Tunnel : ssh -L 8083:${CONTAINER_IP}:8083 <spark-hostname>"
echo "  URL    : http://localhost:8083/"
echo ""
echo "  Left-click  = add to mask"
echo "  Right-click = exclude from mask"
echo "  Scroll      = zoom"
echo "  Middle-drag = pan"
echo ""

python3 "${SCRIPT_DIR}/tools/interactive_editor/serve.py" "$IMAGE"
