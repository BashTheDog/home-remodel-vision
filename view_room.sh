#!/usr/bin/env bash
# view_room.sh <room_name> [port]
#
# Serves a trained 3DGS splat via the web viewer.
#
# Args:
#   room_name : name used when process_room.sh was run (e.g. kitchen)
#   port      : TCP port (default: 8080)
#
# Access from Mac:
#   1. SSH tunnel: ssh -L 8080:localhost:8080 <spark-hostname>
#   2. Open: http://localhost:8080/
#
# Example:
#   bash view_room.sh bedroom1
#   bash view_room.sh kitchen 8081

set -euo pipefail

ROOM="${1:-bedroom1}"
PORT="${2:-8080}"

PLY="/project/outputs/${ROOM}_3dgs/${ROOM}.ply"

if [[ ! -f "$PLY" ]]; then
    echo "ERROR: No splat found at ${PLY}"
    echo "  Run:  bash /project/process_room.sh <start> <end> ${ROOM}"
    exit 1
fi

echo ""
echo "  Room      : ${ROOM}"
echo "  PLY       : ${PLY}"
echo "  Port      : ${PORT}"
echo ""
echo "  Mac SSH tunnel:  ssh -L ${PORT}:localhost:${PORT} <spark-hostname>"
echo "  Then open:       http://localhost:${PORT}/"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

VIEWER_PORT="$PORT" python3 /project/tools/viewer/serve.py "$ROOM"
