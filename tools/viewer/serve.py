#!/usr/bin/env python3
"""
HTTP server for the 3DGS viewer.

Usage:
    python3 serve.py [room_name]

Serves /project/outputs/<room_name>_3dgs/<room_name>.ply via the viewer.
Defaults to bedroom1 if no room_name given.
"""
import http.server, os, sys, signal

PORT      = int(os.environ.get("VIEWER_PORT", "8080"))
ROOM_NAME = sys.argv[1] if len(sys.argv) > 1 else "bedroom1"

VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
PLY_SRC    = f"/project/outputs/{ROOM_NAME}_3dgs/{ROOM_NAME}.ply"
PLY_LINK   = os.path.join(VIEWER_DIR, "scene.ply")

if not os.path.exists(PLY_SRC):
    sys.exit(f"ERROR: PLY not found: {PLY_SRC}\nRun process_room.sh first.")

# Symlink the room's PLY to a fixed name the HTML requests
if os.path.islink(PLY_LINK) or os.path.exists(PLY_LINK):
    os.remove(PLY_LINK)
os.symlink(PLY_SRC, PLY_LINK)
print(f"Linked {PLY_SRC} → {PLY_LINK}")

os.chdir(VIEWER_DIR)


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ''
        if not any(first.endswith(ext) for ext in ('.js', '.wasm', '.ply')):
            super().log_message(fmt, *args)


def run():
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), CORSHandler)
    print(f"\n  3DGS Viewer  →  http://0.0.0.0:{PORT}/")
    print(f"  Room         →  {ROOM_NAME}")
    print(f"  PLY          →  {PLY_SRC}")
    print(f"  Mac tunnel   →  ssh -L {PORT}:localhost:{PORT} <spark-hostname>")
    print(f"  Then open    →  http://localhost:{PORT}/\n")

    def _stop(sig, frame):
        print("\nShutting down.")
        server.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
