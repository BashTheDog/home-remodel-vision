#!/usr/bin/env python3
"""
Before/after edit viewer.

Usage:
    python3 serve.py <before_image> <after_image> <mask_image>

Serves a drag-to-compare viewer on port 8082 (0.0.0.0).
"""
import http.server, os, sys, signal, socket

PORT       = int(os.environ.get("EDIT_PORT", "8082"))
VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 4:
    sys.exit("Usage: serve.py <before_image> <after_image> <mask_image>")

before_path = os.path.abspath(sys.argv[1])
after_path  = os.path.abspath(sys.argv[2])
mask_path   = os.path.abspath(sys.argv[3])

for label, path in [("before", before_path), ("after", after_path), ("mask", mask_path)]:
    if not os.path.exists(path):
        sys.exit(f"ERROR: {label} image not found: {path}")

# Symlink images under fixed names
for name, src in [("before.jpg", before_path), ("after.jpg", after_path), ("mask.jpg", mask_path)]:
    link = os.path.join(VIEWER_DIR, name)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    os.symlink(src, link)

os.chdir(VIEWER_DIR)


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ""
        if not any(first.endswith(ext) for ext in (".jpg", ".jpeg", ".png")):
            super().log_message(fmt, *args)


def container_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def run():
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), CORSHandler)
    ip = container_ip()
    print(f"\n  Edit Viewer  →  http://0.0.0.0:{PORT}/")
    print(f"  Before       →  {before_path}")
    print(f"  After        →  {after_path}")
    print(f"  Mask         →  {mask_path}")
    print(f"  Container IP →  {ip}")
    print(f"  Mac tunnel   →  ssh -L {PORT}:{ip}:{PORT} <spark-hostname>")
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
