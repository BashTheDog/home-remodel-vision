#!/usr/bin/env bash
# Usage: ./tour_room.sh <room_name> [port]
set -euo pipefail

ROOM="${1:-bedroom1}"
PORT="${2:-8081}"   # default 8081 to avoid conflict with view_room.sh

FRAMES_DIR="/project/data/frames/${ROOM}"
if [[ ! -d "$FRAMES_DIR" ]]; then
    echo "ERROR: No frames found at $FRAMES_DIR"
    echo "Run: ./process_room.sh <start> <end> $ROOM"
    exit 1
fi

COUNT=$(find "$FRAMES_DIR" -name '*.jpg' | wc -l)
echo "Room    : $ROOM"
echo "Frames  : $COUNT"
echo "Port    : $PORT"
echo ""
echo "  SSH tunnel  : ssh -L ${PORT}:172.18.0.3:${PORT} pharn@192.168.0.10"
echo "  Open        : http://localhost:${PORT}/"
echo ""

VIEWER_PORT="$PORT" python3 /project/tools/tour/serve.py "$ROOM" "$PORT"
