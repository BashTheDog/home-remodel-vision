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

Outputs to /project/outputs/edited/<output_name>_{before,mask,after}.jpg
"""

import argparse
import os
import sys
import warnings
import logging

import cv2
import numpy as np
from PIL import Image

# ── suppress noisy but harmless warnings ──────────────────────────────────────
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── paths ─────────────────────────────────────────────────────────────────────
SAM2_CHECKPOINT  = "/project/models/sam2/sam2.1_hiera_small.pt"
SAM2_CONFIG      = "configs/sam2.1/sam2.1_hiera_s.yaml"
GDINO_CONFIG     = "/project/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT = "/project/models/groundingdino/groundingdino_swint_ogc.pth"
OUTPUT_DIR       = "/project/outputs/edited"

MASK_DILATE_PX  = 12    # pixels to grow the mask before inpainting
GDINO_BOX_THR   = 0.35
GDINO_TEXT_THR  = 0.25


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
    """Save a red-tinted overlay of the mask on the original image."""
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

    model = load_model(GDINO_CONFIG, GDINO_CHECKPOINT, device="cuda")
    image_source, image_tensor = load_image(image_path)   # source is H×W×3 uint8 RGB
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
        sys.exit(f"ERROR: GroundingDINO found nothing matching '{prompt}'. "
                 "Try a different prompt or lower --box-threshold.")

    # boxes are [cx, cy, w, h] normalised; pick highest-confidence
    best = logits.argmax().item()
    cx, cy, bw, bh = boxes[best].tolist()
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    box_xyxy = [x1, y1, x2, y2]

    print(f"  [detect] best match: '{phrases[best]}' "
          f"conf={logits[best]:.2f}  box={[round(v) for v in box_xyxy]}")

    # Free GPU memory before loading SAM2
    del model
    import torch; torch.cuda.empty_cache()

    return image_source, box_xyxy


# ── segmentation: SAM2 ────────────────────────────────────────────────────────

def segment_with_sam2(image_rgb, *, box=None, points=None):
    """
    image_rgb : H×W×3 uint8 numpy array
    box       : [x1, y1, x2, y2] in pixels  (used for prompt-mode AND manual-box mode)
    points    : [[x, y], ...]  (manual click-point mode)
    Returns   : H×W uint8 mask (0/255)
    """
    print("  [segment] SAM2 predicting mask…")
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
    predictor = SAM2ImagePredictor(model)

    with torch.inference_mode():
        predictor.set_image(image_rgb)

        if box is not None:
            box_np = np.array(box, dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box_np,
                multimask_output=True,
            )
        else:
            pts = np.array(points, dtype=np.float32)
            labels = np.ones(len(pts), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=labels,
                multimask_output=True,
            )

    best = scores.argmax()
    mask = (masks[best] > 0).astype(np.uint8) * 255
    print(f"  [segment] mask coverage: {mask.sum() // 255:,} px  (score={scores[best]:.3f})")

    del predictor, model
    import torch; torch.cuda.empty_cache()

    return mask


# ── inpainting: LaMa ─────────────────────────────────────────────────────────

def inpaint_lama(image_rgb, mask_uint8):
    """
    Crop to the mask bounding box (+ margin), inpaint that region,
    paste back.  Avoids PyTorch 32-bit index limits on large panoramas.
    """
    print("  [inpaint] LaMa (cropped region)…")
    from simple_lama_inpainting import SimpleLama

    h, w = image_rgb.shape[:2]

    # Find mask bounding box
    rows = np.where(mask_uint8.any(axis=1))[0]
    cols = np.where(mask_uint8.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        print("  WARNING: empty mask, returning original image")
        return image_rgb.copy()

    margin = 200   # context pixels around the mask
    y1 = max(0,    rows[0]  - margin)
    y2 = min(h,    rows[-1] + margin + 1)
    x1 = max(0,    cols[0]  - margin)
    x2 = min(w,    cols[-1] + margin + 1)

    print(f"  [inpaint] crop region: ({x1},{y1})→({x2},{y2})  "
          f"size={x2-x1}×{y2-y1}")

    crop_img  = image_rgb[y1:y2, x1:x2]
    crop_mask = mask_uint8[y1:y2, x1:x2]

    lama = SimpleLama()
    img_pil  = Image.fromarray(crop_img)
    mask_pil = Image.fromarray(crop_mask).convert("L")

    crop_h, crop_w = crop_img.shape[:2]
    inpainted_crop = np.array(lama(img_pil, mask_pil).convert("RGB"))
    # LaMa may pad to a stride multiple; crop back to original region size
    inpainted_crop = inpainted_crop[:crop_h, :crop_w]

    # Paste back
    result = image_rgb.copy()
    result[y1:y2, x1:x2] = inpainted_crop
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Remove an object and inpaint the gap.")
    p.add_argument("input",       help="Input image path")
    p.add_argument("output_name", help="Output basename (no extension)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prompt", metavar="TEXT",
                     help='Text description of object to remove, e.g. "couch"')
    grp.add_argument("--point",  metavar="X,Y",
                     help="Click point on the object, e.g. 512,384")
    grp.add_argument("--box",    metavar="X1,Y1,X2,Y2",
                     help="Rough bounding box, e.g. 100,200,400,600")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = os.path.join(OUTPUT_DIR, args.output_name)

    print(f"\n=== Stage 2: Object Removal ===")
    print(f"  Input  : {args.input}")
    print(f"  Output : {prefix}_after.jpg\n")

    # ── detection / selection ─────────────────────────────────────────────────
    image_rgb = load_image_rgb(args.input)
    h, w = image_rgb.shape[:2]
    print(f"  Image  : {w}×{h}")

    if args.prompt:
        image_source, box = detect_with_grounding_dino(args.input, args.prompt)
        # Use image_source from GroundingDINO (it may have been resized/normalised
        # for model input); use our original image_rgb for actual processing
        mask = segment_with_sam2(image_rgb, box=box)

    elif args.point:
        x, y = [float(v) for v in args.point.split(",")]
        print(f"  [select] manual point: ({x}, {y})")
        mask = segment_with_sam2(image_rgb, points=[[x, y]])

    else:  # --box
        vals = [float(v) for v in args.box.split(",")]
        box = vals[:4]
        print(f"  [select] manual box: {box}")
        mask = segment_with_sam2(image_rgb, box=box)

    # ── dilate mask ───────────────────────────────────────────────────────────
    mask = dilate_mask(mask, MASK_DILATE_PX)

    # ── save before + mask ────────────────────────────────────────────────────
    before_path = f"{prefix}_before.jpg"
    mask_path   = f"{prefix}_mask.jpg"
    after_path  = f"{prefix}_after.jpg"

    Image.fromarray(image_rgb).save(before_path, quality=92)
    save_mask_viz(mask, image_rgb, mask_path)
    print(f"  Saved  : {before_path}")
    print(f"  Saved  : {mask_path}")

    # ── inpaint ───────────────────────────────────────────────────────────────
    result = inpaint_lama(image_rgb, mask)
    Image.fromarray(result).save(after_path, quality=92)
    print(f"  Saved  : {after_path}")

    # ── viewer instructions ───────────────────────────────────────────────────
    ip   = container_ip()
    port = 8082
    print(f"\n=== Done ===")
    print(f"  Before : {before_path}")
    print(f"  Mask   : {mask_path}")
    print(f"  After  : {after_path}")
    print(f"\nTo view before/after in browser:")
    print(f"  python3 /project/tools/edit_viewer/serve.py {before_path} {after_path} {mask_path}")
    print(f"  Tunnel : ssh -L 8082:{ip}:8082 <spark-hostname>")
    print(f"  URL    : http://localhost:8082/")


if __name__ == "__main__":
    main()
