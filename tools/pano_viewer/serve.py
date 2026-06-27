#!/usr/bin/env python3
"""
Panorama viewer server.

Usage:
    python3 serve.py <image_path>

Serves a zoomable/pannable browser viewer for a large panorama JPEG.
Binds to 0.0.0.0 so it's reachable via SSH tunnel from a Mac.
"""
import http.server, os, sys, signal, socket

PORT       = int(os.environ.get("PANO_PORT", "8081"))
VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) < 2:
    sys.exit("Usage: serve.py <image_path>")

img_path = os.path.abspath(sys.argv[1])
if not os.path.exists(img_path):
    sys.exit(f"ERROR: image not found: {img_path}")

# Symlink image into viewer dir under a fixed name so HTML can fetch it
img_link = os.path.join(VIEWER_DIR, "pano.jpg")
if os.path.islink(img_link) or os.path.exists(img_link):
    os.remove(img_link)
os.symlink(img_path, img_link)
print(f"Linked {img_path} → {img_link}")

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
    """Return this container's IP (first non-loopback addr via routing table)."""
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
    print(f"\n  Panorama Viewer  →  http://0.0.0.0:{PORT}/")
    print(f"  Image            →  {img_path}")
    print(f"  Container IP     →  {ip}")
    print(f"  Mac tunnel       →  ssh -L {PORT}:{ip}:{PORT} <spark-hostname>")
    print(f"  Then open        →  http://localhost:{PORT}/\n")

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
