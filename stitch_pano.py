#!/usr/bin/env python3
"""
Vertical panorama stitcher — upper + lower image → one tall composite.

Usage:
    python3 stitch_pano.py <upper> <lower> <output>

Strategy:
  1. Try cv2.Stitcher (SCANS mode — flat projection, good for 2-image vertical).
  2. On failure, fall back to SIFT feature matching → homography → feathered blend.

Detection is done at 1/4 scale for speed; homography is scaled back to full res.
"""

import sys
import os
import numpy as np
import cv2

DETECT_SCALE = 0.25   # fraction of original for feature detection
JPEG_QUALITY = 90
MIN_MATCHES   = 15    # minimum good feature matches to trust homography


def load(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"ERROR: cannot read image: {path}")
    return img


def try_stitcher(upper, lower):
    """Attempt cv2.Stitcher in SCANS mode (flat, no projection warp)."""
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch([upper, lower])
    if status == cv2.Stitcher_OK:
        print("Stitcher succeeded.")
        return pano
    codes = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "need more images",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography failed",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera adjust failed",
    }
    reason = codes.get(status, f"code {status}")
    print(f"Stitcher failed ({reason}); falling back to feature matching.")
    return None


def detect_and_match(upper, lower):
    """SIFT at reduced scale, FLANN match, ratio test."""
    def downscale(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)),
                          interpolation=cv2.INTER_AREA)

    u_small = downscale(upper)
    l_small = downscale(lower)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp_u, des_u = sift.detectAndCompute(cv2.cvtColor(u_small, cv2.COLOR_BGR2GRAY), None)
    kp_l, des_l = sift.detectAndCompute(cv2.cvtColor(l_small, cv2.COLOR_BGR2GRAY), None)
    print(f"Keypoints — upper: {len(kp_u)}, lower: {len(kp_l)}")

    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(des_l, des_u, k=2)

    # Lowe ratio test
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f"Good matches after ratio test: {len(good)}")

    if len(good) < MIN_MATCHES:
        sys.exit(f"ERROR: only {len(good)} good matches (need {MIN_MATCHES}). "
                 "Check that the images actually overlap.")

    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good])
    pts_u = np.float32([kp_u[m.trainIdx].pt for m in good])
    return pts_l, pts_u


def build_composite(upper, lower, pts_l, pts_u):
    """
    Find homography H (lower small-scale → upper small-scale), scale to full res,
    warp lower, place both on a canvas, feather-blend the overlap.
    """
    H_small, mask = cv2.findHomography(pts_l, pts_u, cv2.RANSAC, 5.0)
    inliers = int(mask.sum())
    print(f"Homography inliers: {inliers}")
    if inliers < MIN_MATCHES:
        sys.exit(f"ERROR: only {inliers} homography inliers — images may not overlap enough.")

    # Scale H from detection scale → full resolution
    S = 1.0 / DETECT_SCALE
    T = np.array([[S, 0, 0], [0, S, 0], [0, 0, 1]], dtype=np.float64)
    T_inv = np.array([[DETECT_SCALE, 0, 0], [0, DETECT_SCALE, 0], [0, 0, 1]], dtype=np.float64)
    H = T @ H_small @ T_inv   # H at full resolution

    # Find where lower image corners land in upper coordinate space
    h_u, w_u = upper.shape[:2]
    h_l, w_l = lower.shape[:2]
    corners_l = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2)
    corners_in_upper = cv2.perspectiveTransform(corners_l, H).reshape(-1, 2)

    all_x = np.concatenate([[0, w_u], corners_in_upper[:, 0]])
    all_y = np.concatenate([[0, h_u], corners_in_upper[:, 1]])
    min_x, min_y = int(np.floor(all_x.min())), int(np.floor(all_y.min()))
    max_x, max_y = int(np.ceil(all_x.max())),  int(np.ceil(all_y.max()))

    # Translation to keep everything in positive coordinates
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0
    canvas_w = max_x + tx
    canvas_h = max_y + ty

    print(f"Canvas size: {canvas_w}×{canvas_h}  (offset tx={tx}, ty={ty})")

    T_canvas = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_canvas = T_canvas @ H   # lower → canvas

    # Warp lower into canvas space
    lower_warped = cv2.warpPerspective(lower, H_canvas, (canvas_w, canvas_h))

    # Build a mask of where lower has real pixels (non-black after warp)
    lower_mask_warped = cv2.warpPerspective(
        np.ones((h_l, w_l), dtype=np.float32), H_canvas, (canvas_w, canvas_h))

    # Place upper on canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    upper_mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    canvas[ty:ty+h_u, tx:tx+w_u] = upper.astype(np.float32)
    upper_mask[ty:ty+h_u, tx:tx+w_u] = 1.0

    # Feathered blend in overlap region
    overlap = (upper_mask > 0) & (lower_mask_warped > 0.5)
    print(f"Overlap pixels: {overlap.sum():,}")

    if overlap.sum() > 0:
        # Vertical gradient in the overlap band: upper fades out top→bottom
        overlap_rows = np.where(overlap.any(axis=1))[0]
        row_top, row_bot = overlap_rows[0], overlap_rows[-1]
        span = max(row_bot - row_top, 1)

        weight_u = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight_u[ty:ty+h_u, tx:tx+w_u] = upper_mask[ty:ty+h_u, tx:tx+w_u]

        # Ramp upper weight from 1 → 0 over the overlap band
        for r in range(row_top, row_bot + 1):
            alpha = 1.0 - (r - row_top) / span   # 1.0 at top, 0.0 at bottom
            weight_u[r, :] *= alpha

        # Outside overlap: upper gets full weight where it exists, lower elsewhere
        weight_l = np.zeros_like(weight_u)
        weight_l[lower_mask_warped > 0.5] = 1.0
        weight_l = weight_l * (1.0 - weight_u)
        weight_l[~overlap] = lower_mask_warped[~overlap]   # lower-only regions = full weight

        # Normalize (avoid div-by-zero)
        total = weight_u + weight_l
        total = np.where(total == 0, 1, total)
        w_u_3 = (weight_u / total)[..., np.newaxis]
        w_l_3 = (weight_l / total)[..., np.newaxis]

        result = canvas * w_u_3 + lower_warped.astype(np.float32) * w_l_3
    else:
        # No detected overlap — just place both (shouldn't happen with good homography)
        print("WARNING: no pixel overlap detected; placing images without blending.")
        result = canvas.copy()
        lower_px = lower_mask_warped > 0.5
        result[lower_px] = lower_warped.astype(np.float32)[lower_px]

    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: stitch_pano.py <upper> <lower> <output>")

    upper_path, lower_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Loading images…")
    upper = load(upper_path)
    lower = load(lower_path)
    print(f"Upper: {upper.shape[1]}×{upper.shape[0]}  Lower: {lower.shape[1]}×{lower.shape[0]}")

    # 1. Try Stitcher
    result = try_stitcher(upper, lower)

    # 2. Fall back to feature matching
    if result is None:
        print("Running SIFT feature matching…")
        pts_l, pts_u = detect_and_match(upper, lower)
        result = build_composite(upper, lower, pts_l, pts_u)

    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    h, w = result.shape[:2]
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path}  ({w}×{h}, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
