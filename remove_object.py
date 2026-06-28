#!/usr/bin/env python3
"""
Stage 2 — Object removal via detection → segmentation → inpainting.

Usage:
  # Text prompt (GroundingDINO → SAM2 → LaMa):
  python3 remove_object.py <input> <output_name> --prompt "couch"

  # Click point (SAM2 only):
  python3 remove_object.py <input> <output_name> --point 512,384

  # Bounding box (SAM2 only):
  python3 remove_object.py <input> <output_name> --box 100,200,400,600

  # Use small model (faster, lower quality):
  python3 remove_object.py <input> <output_name> --prompt "couch" --model small

Outputs to /project/outputs/edited/<output_name>_{before,mask,after}.jpg

Key design: SAM2 is always run on a crop around the object, not the full
image.  On an 11 K-pixel panorama, SAM2 would otherwise downsample 11× and
lose all fine edge detail.  Cropping gives it full resolution on the target.
"""

import argparse
import os
import sys
import warnings
import logging

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── model paths ───────────────────────────────────────────────────────────────
MODELS = {
    "large": {
        "checkpoint": "/project/models/sam2/sam2.1_hiera_large.pt",
        "config":     "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
    "small": {
        "checkpoint": "/project/models/sam2/sam2.1_hiera_small.pt",
        "config":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
}
DEFAULT_MODEL   = "large"
GDINO_CONFIG    = "/project/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_WEIGHTS   = "/project/models/groundingdino/groundingdino_swint_ogc.pth"
OUTPUT_DIR      = "/project/outputs/edited"

GDINO_BOX_THR   = 0.30   # lower = catch more (at risk of false positives)
GDINO_TEXT_THR  = 0.25
MASK_DILATE_PX  = 8      # pixels to grow mask edges before inpainting
BOX_PAD_FRAC    = 0.35   # fraction of box size to add as padding for SAM2 crop
POINT_CROP_HALF = 2000   # half-size of crop window in point mode (pixels)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"ERROR: cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def dilate_mask(mask_uint8, px):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask_uint8, k)


def save_mask_viz(mask_uint8, img_rgb, out_path):
    overlay = img_rgb.copy()
    overlay[mask_uint8 > 0] = [220, 30, 30]
    blended = (img_rgb * 0.5 + overlay * 0.5).astype(np.uint8)
    Image.fromarray(blended).save(out_path, quality=90)


def container_ip():
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ── detection: GroundingDINO ──────────────────────────────────────────────────

def detect_with_grounding_dino(image_path, prompt):
    print(f"  [detect] GroundingDINO: '{prompt}'")
    from groundingdino.util.inference import load_model, load_image, predict

    model = load_model(GDINO_CONFIG, GDINO_WEIGHTS, device="cuda")
    image_source, image_tensor = load_image(image_path)
    h, w = image_source.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=GDINO_BOX_THR,
        text_threshold=GDINO_TEXT_THR,
        device="cuda",
    )

    if len(boxes) == 0:
        sys.exit(
            f"ERROR: GroundingDINO found nothing matching '{prompt}'.\n"
            f"  Try a broader word, or use --point / --box for manual selection.\n"
            f"  (box_threshold={GDINO_BOX_THR}; lower it in the script to catch weaker matches)"
        )

    # Print all detections so the user can see what was found
    print(f"  [detect] {len(boxes)} detection(s):")
    for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
        cx, cy, bw, bh = box.tolist()
        px1, py1 = int((cx - bw/2)*w), int((cy - bh/2)*h)
        px2, py2 = int((cx + bw/2)*w), int((cy + bh/2)*h)
        marker = " ← best" if i == logits.argmax().item() else ""
        print(f"    [{i}] '{phrase}'  conf={logit:.2f}  box=[{px1},{py1},{px2},{py2}]{marker}")

    best = logits.argmax().item()
    cx, cy, bw, bh = boxes[best].tolist()
    box_xyxy = [
        (cx - bw/2) * w, (cy - bh/2) * h,
        (cx + bw/2) * w, (cy + bh/2) * h,
    ]

    import torch; del model; torch.cuda.empty_cache()
    return box_xyxy


# ── segmentation: SAM2 on a crop ─────────────────────────────────────────────

def segment_with_sam2(image_rgb, *, box=None, points=None, model_key=DEFAULT_MODEL):
    """
    Crops to the object region first so SAM2 receives full-resolution detail,
    then projects the resulting mask back to original image coordinates.

    image_rgb : H×W×3 uint8
    box       : [x1,y1,x2,y2] in original pixels
    points    : [[x,y], ...] in original pixels
    Returns   : H×W uint8 mask (0/255) in original image space
    """
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    h_full, w_full = image_rgb.shape[:2]
    cfg = MODELS[model_key]

    # ── compute crop bounds ───────────────────────────────────────────────────
    if box is not None:
        x1, y1, x2, y2 = [float(v) for v in box]
        pad_x = max(150, int((x2 - x1) * BOX_PAD_FRAC))
        pad_y = max(150, int((y2 - y1) * BOX_PAD_FRAC))
        cx1 = max(0,       int(x1 - pad_x))
        cy1 = max(0,       int(y1 - pad_y))
        cx2 = min(w_full,  int(x2 + pad_x))
        cy2 = min(h_full,  int(y2 + pad_y))
    else:
        # Point mode: fixed window around the click
        px, py = int(points[0][0]), int(points[0][1])
        cx1 = max(0,      px - POINT_CROP_HALF)
        cy1 = max(0,      py - POINT_CROP_HALF)
        cx2 = min(w_full, px + POINT_CROP_HALF)
        cy2 = min(h_full, py + POINT_CROP_HALF)

    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    print(f"  [segment] SAM2 ({model_key}) on crop "
          f"({cx1},{cy1})→({cx2},{cy2})  {crop_w}×{crop_h}px")

    crop = image_rgb[cy1:cy2, cx1:cx2]

    # ── translate coords into crop space ─────────────────────────────────────
    if box is not None:
        prompt_box = np.array(
            [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], dtype=np.float32
        )
    else:
        pts_crop   = np.array([[p[0]-cx1, p[1]-cy1] for p in points], dtype=np.float32)
        pts_labels = np.ones(len(pts_crop), dtype=np.int32)

    # ── run SAM2 ──────────────────────────────────────────────────────────────
    model     = build_sam2(cfg["config"], cfg["checkpoint"], device="cuda")
    predictor = SAM2ImagePredictor(model)

    with torch.inference_mode():
        predictor.set_image(crop)

        if box is not None:
            masks, scores, _ = predictor.predict(
                box=prompt_box,
                multimask_output=True,
            )
        else:
            masks, scores, _ = predictor.predict(
                point_coords=pts_crop,
                point_labels=pts_labels,
                multimask_output=True,
            )

    # Report all three candidate masks so quality is visible in logs
    for i, s in enumerate(scores):
        print(f"    mask[{i}] score={s:.3f}  coverage={int(masks[i].sum()):,}px")

    best       = int(scores.argmax())
    mask_crop  = (masks[best] > 0).astype(np.uint8) * 255
    print(f"  [segment] using mask[{best}]  score={scores[best]:.3f}")

    del predictor, model
    torch.cuda.empty_cache()

    # ── project back to full image ────────────────────────────────────────────
    mask_full = np.zeros((h_full, w_full), dtype=np.uint8)
    mask_full[cy1:cy2, cx1:cx2] = mask_crop
    return mask_full


# ── inpainting: LaMa on a crop ───────────────────────────────────────────────

def inpaint_lama(image_rgb, mask_uint8):
    """
    Crop to mask bounding box + margin, inpaint, paste back.
    Avoids PyTorch 32-bit index limits on large panoramas.
    """
    print("  [inpaint] LaMa…")
    from simple_lama_inpainting import SimpleLama

    h, w = image_rgb.shape[:2]
    rows = np.where(mask_uint8.any(axis=1))[0]
    cols = np.where(mask_uint8.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        print("  WARNING: empty mask — returning original")
        return image_rgb.copy()

    margin = 200
    y1 = max(0, rows[0]  - margin)
    y2 = min(h, rows[-1] + margin + 1)
    x1 = max(0, cols[0]  - margin)
    x2 = min(w, cols[-1] + margin + 1)
    print(f"  [inpaint] region ({x1},{y1})→({x2},{y2})  {x2-x1}×{y2-y1}px")

    crop_img  = image_rgb[y1:y2, x1:x2]
    crop_mask = mask_uint8[y1:y2, x1:x2]
    crop_h, crop_w = crop_img.shape[:2]

    lama = SimpleLama()
    out  = np.array(lama(Image.fromarray(crop_img),
                         Image.fromarray(crop_mask).convert("L")).convert("RGB"))
    out  = out[:crop_h, :crop_w]   # LaMa pads to stride multiple; trim back

    result = image_rgb.copy()
    result[y1:y2, x1:x2] = out
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input",       help="Input image path")
    p.add_argument("output_name", help="Output basename (no extension)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prompt", metavar="TEXT",
                     help='Text description, e.g. "couch"')
    grp.add_argument("--point",  metavar="X,Y",
                     help="Click point on the object, e.g. 512,384")
    grp.add_argument("--box",    metavar="X1,Y1,X2,Y2",
                     help="Rough bounding box, e.g. 100,200,400,600")
    p.add_argument("--model", choices=["large", "small"], default=DEFAULT_MODEL,
                   help="SAM2 model size (default: large)")
    p.add_argument("--no-inpaint", action="store_true",
                   help="Stop after masking; skip inpainting (useful for mask QA)")
    return p.parse_args()


def main():
    args = parse_args()

    # Check model exists
    ckpt = MODELS[args.model]["checkpoint"]
    if not os.path.exists(ckpt):
        sys.exit(
            f"ERROR: SAM2 {args.model} model not found at {ckpt}\n"
            f"  Run: wget -O {ckpt} "
            f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_{'large' if args.model=='large' else 'small'}.pt"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = os.path.join(OUTPUT_DIR, args.output_name)

    print(f"\n=== Stage 2: Object Removal ===")
    print(f"  Input  : {args.input}")
    print(f"  Model  : SAM2 {args.model}")
    print(f"  Output : {prefix}_after.jpg\n")

    image_rgb = load_image_rgb(args.input)
    h, w = image_rgb.shape[:2]
    print(f"  Image  : {w}×{h}")

    # ── detect / select ───────────────────────────────────────────────────────
    if args.prompt:
        box = detect_with_grounding_dino(args.input, args.prompt)
        mask = segment_with_sam2(image_rgb, box=box, model_key=args.model)

    elif args.point:
        x, y = [float(v) for v in args.point.split(",")]
        print(f"  [select] point ({x}, {y})")
        mask = segment_with_sam2(image_rgb, points=[[x, y]], model_key=args.model)

    else:
        vals = [float(v) for v in args.box.split(",")]
        print(f"  [select] box {vals}")
        mask = segment_with_sam2(image_rgb, box=vals[:4], model_key=args.model)

    mask = dilate_mask(mask, MASK_DILATE_PX)

    # ── save before + mask ────────────────────────────────────────────────────
    before_path = f"{prefix}_before.jpg"
    mask_path   = f"{prefix}_mask.jpg"
    after_path  = f"{prefix}_after.jpg"

    Image.fromarray(image_rgb).save(before_path, quality=92)
    save_mask_viz(mask, image_rgb, mask_path)
    print(f"  Saved  : {before_path}")
    print(f"  Saved  : {mask_path}")

    if args.no_inpaint:
        print("\n  --no-inpaint set; stopping after mask.")
        print(f"  View mask: open {mask_path}")
        return

    # ── inpaint ───────────────────────────────────────────────────────────────
    result = inpaint_lama(image_rgb, mask)
    Image.fromarray(result).save(after_path, quality=92)
    print(f"  Saved  : {after_path}")

    ip = container_ip()
    print(f"\n=== Done ===")
    print(f"  python3 /project/tools/edit_viewer/serve.py {before_path} {after_path} {mask_path}")
    print(f"  Tunnel : ssh -L 8082:{ip}:8082 <spark-hostname>")
    print(f"  URL    : http://localhost:8082/")


if __name__ == "__main__":
    main()
