#!/usr/bin/env python3
"""
Interactive object removal editor.

Usage:
    python3 serve.py <image_path>

Left-click  → positive SAM2 point (add to mask)
Right-click → negative SAM2 point (exclude from mask)
Remove      → LaMa inpaints the masked region
"""

import http.server
import json
import base64
import io
import os
import sys
import signal
import socket
import threading
import warnings
import logging

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PORT       = int(os.environ.get("EDITOR_PORT", "8083"))
VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))

SAM2_CHECKPOINT = "/project/models/sam2/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
MASK_DILATE_PX  = 8
POINT_CROP_HALF = 2000   # half-window around click for SAM2 crop
LAMA_MARGIN     = 200

# ── argument parsing ──────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    sys.exit("Usage: serve.py <image_path>")
IMAGE_PATH = os.path.abspath(sys.argv[1])
if not os.path.exists(IMAGE_PATH):
    sys.exit(f"ERROR: image not found: {IMAGE_PATH}")

RESULT_PATH = os.path.join(VIEWER_DIR, "result.jpg")

# ── shared state ──────────────────────────────────────────────────────────────
_lock         = threading.Lock()
_image_rgb    = None   # full-res numpy H×W×3
_predictor    = None
_crop_bounds  = None   # (cx1, cy1, cx2, cy2) currently encoded in predictor
_current_mask = None   # full-res H×W uint8 mask (0/255)


# ── model init (called at startup) ───────────────────────────────────────────

def load_sam2():
    global _predictor, _image_rgb
    print(f"Loading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    _image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = _image_rgb.shape[:2]
    print(f"Image: {w}×{h}")

    print("Loading SAM2 large model (857 MB)…")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model      = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
    _predictor = SAM2ImagePredictor(model)
    print("SAM2 ready.")


# ── segmentation ──────────────────────────────────────────────────────────────

def _set_crop(cx1, cy1, cx2, cy2):
    """Encode the crop into the SAM2 predictor (slow — encodes image features)."""
    global _crop_bounds
    import torch
    crop = _image_rgb[cy1:cy2, cx1:cx2]
    with torch.inference_mode():
        _predictor.set_image(crop)
    _crop_bounds = (cx1, cy1, cx2, cy2)


def do_segment(points, labels):
    """
    points : [[x, y], ...] in original image coordinates
    labels : [1|0, ...]  (1=add, 0=exclude)
    Returns: (mask_png_bytes, score, mask_full_uint8)
    """
    import torch

    h_full, w_full = _image_rgb.shape[:2]

    # Centre crop on positive points
    pos = [p for p, l in zip(points, labels) if l == 1]
    if not pos:
        return None, 0.0, None

    cx = int(np.mean([p[0] for p in pos]))
    cy = int(np.mean([p[1] for p in pos]))

    # Re-encode if crop doesn't yet exist or no longer contains all points
    need_encode = _crop_bounds is None
    if not need_encode:
        bcx1, bcy1, bcx2, bcy2 = _crop_bounds
        need_encode = any(
            not (bcx1 <= p[0] <= bcx2 and bcy1 <= p[1] <= bcy2)
            for p in points
        )

    if need_encode:
        new_cx1 = max(0,      cx - POINT_CROP_HALF)
        new_cy1 = max(0,      cy - POINT_CROP_HALF)
        new_cx2 = min(w_full, cx + POINT_CROP_HALF)
        new_cy2 = min(h_full, cy + POINT_CROP_HALF)
        _set_crop(new_cx1, new_cy1, new_cx2, new_cy2)

    bx1, by1, bx2, by2 = _crop_bounds

    pts_crop = np.array([[p[0]-bx1, p[1]-by1] for p in points], dtype=np.float32)
    lbl_arr  = np.array(labels, dtype=np.int32)

    with torch.inference_mode():
        masks, scores, _ = _predictor.predict(
            point_coords=pts_crop,
            point_labels=lbl_arr,
            multimask_output=True,
        )

    best      = int(scores.argmax())
    mask_crop = (masks[best] > 0).astype(np.uint8) * 255

    # Dilate
    k         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE_PX*2+1, MASK_DILATE_PX*2+1))
    mask_crop = cv2.dilate(mask_crop, k)

    # Full-image mask
    mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
    mask_full[by1:by2, bx1:bx2] = mask_crop

    # RGBA overlay PNG (only the crop region — keeps response small)
    rgba                 = np.zeros((*mask_crop.shape, 4), dtype=np.uint8)
    rgba[mask_crop > 0] = [220, 30, 30, 160]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)

    return buf.getvalue(), float(scores[best]), mask_full


# ── inpainting ────────────────────────────────────────────────────────────────

def do_inpaint(mask_full):
    from simple_lama_inpainting import SimpleLama

    h, w  = _image_rgb.shape[:2]
    rows  = np.where(mask_full.any(axis=1))[0]
    cols  = np.where(mask_full.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None

    y1 = max(0, rows[0]  - LAMA_MARGIN)
    y2 = min(h, rows[-1] + LAMA_MARGIN + 1)
    x1 = max(0, cols[0]  - LAMA_MARGIN)
    x2 = min(w, cols[-1] + LAMA_MARGIN + 1)

    crop_img  = _image_rgb[y1:y2, x1:x2]
    crop_mask = mask_full[y1:y2, x1:x2]
    ch, cw    = crop_img.shape[:2]

    lama = SimpleLama()
    out  = np.array(
        lama(Image.fromarray(crop_img), Image.fromarray(crop_mask).convert("L"))
        .convert("RGB")
    )
    out = out[:ch, :cw]

    result = _image_rgb.copy()
    result[y1:y2, x1:x2] = out

    buf = io.BytesIO()
    Image.fromarray(result).save(buf, "JPEG", quality=92)
    return buf.getvalue()


# ── HTTP handler ──────────────────────────────────────────────────────────────

class EditorHandler(http.server.SimpleHTTPRequestHandler):

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, mime, status=200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            path = os.path.join(VIEWER_DIR, "index.html")
            self._send_bytes(open(path, "rb").read(), "text/html")
        elif self.path == "/image":
            self._send_bytes(open(IMAGE_PATH, "rb").read(), "image/jpeg")
        elif self.path == "/result.jpg" and os.path.exists(RESULT_PATH):
            self._send_bytes(open(RESULT_PATH, "rb").read(), "image/jpeg")
        elif self.path == "/status":
            self._send_json({"ready": _predictor is not None})
        else:
            super().do_GET()

    def do_POST(self):
        global _current_mask, _crop_bounds

        if self.path == "/segment":
            data   = self._read_json()
            pts    = data.get("points", [])
            labels = data.get("labels", [])
            if not pts:
                self._send_json({"error": "no points"}, 400)
                return
            with _lock:
                png_bytes, score, mask = do_segment(pts, labels)
            if png_bytes is None:
                self._send_json({"error": "segmentation failed"}, 500)
                return
            with _lock:
                _current_mask = mask
            bx1, by1, bx2, by2 = _crop_bounds
            self._send_json({
                "mask_b64": base64.b64encode(png_bytes).decode(),
                "bounds":   [bx1, by1, bx2, by2],
                "score":    round(score, 3),
                "coverage": int(mask.sum() // 255),
            })

        elif self.path == "/inpaint":
            with _lock:
                mask = _current_mask
            if mask is None:
                self._send_json({"error": "no mask — click the object first"}, 400)
                return
            result_bytes = do_inpaint(mask)
            if result_bytes is None:
                self._send_json({"error": "inpaint failed"}, 500)
                return
            open(RESULT_PATH, "wb").write(result_bytes)
            self._send_json({"ok": True})

        elif self.path == "/reset":
            with _lock:
                _current_mask = None
                _crop_bounds  = None
            self._send_json({"ok": True})

        else:
            self._send_json({"error": "unknown endpoint"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ""
        if not any(first.endswith(e) for e in (".jpg", ".jpeg", ".png", ".js")):
            super().log_message(fmt, *args)


# ── startup ───────────────────────────────────────────────────────────────────

def container_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def run():
    load_sam2()   # blocks until model is loaded

    http.server.ThreadingHTTPServer.allow_reuse_address = True
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), EditorHandler)
    ip = container_ip()
    print(f"\n  Interactive Editor  →  http://0.0.0.0:{PORT}/")
    print(f"  Image              →  {IMAGE_PATH}")
    print(f"  Mac tunnel         →  ssh -L {PORT}:{ip}:{PORT} <spark-hostname>")
    print(f"  Then open          →  http://localhost:{PORT}/\n")

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
