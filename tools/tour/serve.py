#!/usr/bin/env python3
"""
Tour server — serves a frame-by-frame walkthrough of a room.

Usage:
    python3 serve.py <room_name> [port]

Frames served from /project/data/frames/<room_name>/
"""
import http.server, json, os, sys, signal
from pathlib import Path

ROOM   = sys.argv[1] if len(sys.argv) > 1 else "bedroom1"
PORT   = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
PORT   = int(os.environ.get("VIEWER_PORT", PORT))

FRAMES_DIR = Path(f"/project/data/frames/{ROOM}")
TOUR_DIR   = Path(__file__).parent

if not FRAMES_DIR.exists():
    sys.exit(f"ERROR: frames not found at {FRAMES_DIR}\nRun process_room.sh first.")

FRAMES = sorted(f.name for f in FRAMES_DIR.iterdir() if f.suffix == '.jpg')
if not FRAMES:
    sys.exit(f"ERROR: no .jpg frames found in {FRAMES_DIR}")

print(f"\n  Room    : {ROOM}")
print(f"  Frames  : {len(FRAMES)} ({FRAMES_DIR})")
print(f"  Port    : {PORT}")
print(f"  Tunnel  : ssh -L {PORT}:localhost:{PORT} <spark-hostname>")
print(f"  Open    : http://localhost:{PORT}/\n")


class TourHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        p = self.path.split('?')[0]

        if p == '/':
            self._serve_index()
        elif p == '/frames.json':
            self._serve_json(FRAMES)
        elif p.startswith('/frames/'):
            name = p[len('/frames/'):]
            if name in set(FRAMES):
                self._serve_file(FRAMES_DIR / name, 'image/jpeg')
            else:
                self._404()
        else:
            # Static files from the tour directory (CSS, JS if any)
            fp = TOUR_DIR / p.lstrip('/')
            if fp.is_file():
                ct = 'text/html' if fp.suffix == '.html' else 'application/octet-stream'
                self._serve_file(fp, ct)
            else:
                self._404()

    def _serve_index(self):
        html = (TOUR_DIR / 'index.html').read_bytes()
        # Inject room name and frame list into the HTML
        html = html.replace(b'__ROOM__', ROOM.encode())
        html = html.replace(b'__FRAMES_JSON__', json.dumps(FRAMES).encode())
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self._cors()
        self.send_header('Content-Length', str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path, ct):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _404(self):
        self.send_response(404)
        self.end_headers()

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, max-age=3600')

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ''
        if not any(first.endswith(e) for e in ('.jpg', '.jpeg')):
            super().log_message(fmt, *args)


server = http.server.ThreadingHTTPServer(('0.0.0.0', PORT), TourHandler)

def _stop(sig, frame):
    print('\nShutting down.')
    server.shutdown()
    sys.exit(0)
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
