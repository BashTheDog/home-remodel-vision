#!/usr/bin/env bash
# Stage 2 — Object removal
#
# Usage:
#   remove_object.sh <input_image> <output_name> --prompt "couch"
#   remove_object.sh <input_image> <output_name> --point 512,384
#   remove_object.sh <input_image> <output_name> --box 100,200,400,600
#
# Output files: /project/outputs/edited/<output_name>_{before,mask,after}.jpg

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <input_image> <output_name> --prompt TEXT"
  echo "       $0 <input_image> <output_name> --point X,Y"
  echo "       $0 <input_image> <output_name> --box X1,Y1,X2,Y2"
  exit 1
fi

INPUT="$1"; shift
NAME="$1";  shift
# Remaining args (--prompt / --point / --box) passed through

if [[ "$INPUT" != /* ]]; then INPUT="${SCRIPT_DIR}/${INPUT}"; fi

echo "=== Object Removal ==="
echo "  Input  : $INPUT"
echo "  Name   : $NAME"
echo "  Mode   : $*"
echo ""

python3 "${SCRIPT_DIR}/remove_object.py" "$INPUT" "$NAME" "$@"

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

BEFORE="/project/outputs/edited/${NAME}_before.jpg"
AFTER="/project/outputs/edited/${NAME}_after.jpg"
MASK="/project/outputs/edited/${NAME}_mask.jpg"

echo ""
echo "To view before/after:"
echo "  python3 ${SCRIPT_DIR}/tools/edit_viewer/serve.py $BEFORE $AFTER $MASK"
echo "  Tunnel : ssh -L 8082:${CONTAINER_IP}:8082 <spark-hostname>"
echo "  URL    : http://localhost:8082/"
