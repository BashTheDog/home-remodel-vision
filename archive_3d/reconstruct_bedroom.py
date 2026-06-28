#!/usr/bin/env python3
"""
reconstruct_bedroom.py
First reconstruction test — DUSt3R on the bedroom1 frame set.

Run inside the Workbench container (nvwb attach):
    python3 /project/reconstruct_bedroom.py

Outputs a colored point cloud at /project/outputs/bedroom1.ply
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, "/project/repos/dust3r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/project/data/frames/bedroom1",
                    help="directory of input frames")
    ap.add_argument("--out", default="/project/outputs/bedroom1.ply",
                    help="output .ply path")
    ap.add_argument("--n", type=int, default=30,
                    help="number of frames to subsample for matching")
    ap.add_argument("--niter", type=int, default=300,
                    help="global alignment iterations")
    args = ap.parse_args()

    import torch
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    device = "cuda"

    # ── Select frames ────────────────────────────────────────────────
    all_frames = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if len(all_frames) == 0:
        sys.exit(f"ERROR: no .jpg frames found in {args.frames}")

    if len(all_frames) > args.n:
        idx = np.linspace(0, len(all_frames) - 1, args.n).astype(int)
        frames = [all_frames[i] for i in idx]
    else:
        frames = all_frames
    print(f"[1/6] Using {len(frames)} of {len(all_frames)} frames")

    # ── Load model (downloads weights on first run, ~2 GB) ───────────
    print("[2/6] Loading DUSt3R model (first run downloads weights)...")
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    # ── Load images ──────────────────────────────────────────────────
    print("[3/6] Loading images at 512px...")
    images = load_images(frames, size=512)

    # ── Pairwise inference ───────────────────────────────────────────
    print("[4/6] Building pairs and running inference...")
    pairs = make_pairs(images, scene_graph="complete",
                       prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)

    # ── Global alignment ─────────────────────────────────────────────
    print(f"[5/6] Global alignment ({args.niter} iters, takes a few min)...")
    scene = global_aligner(output, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=args.niter,
                                   schedule="cosine", lr=0.01)

    # ── Extract point cloud ──────────────────────────────────────────
    print("[6/6] Extracting and writing point cloud...")
    pts3d = scene.get_pts3d()
    imgs = scene.imgs
    masks = scene.get_masks()

    all_pts, all_col = [], []
    for pts, img, msk in zip(pts3d, imgs, masks):
        pts = pts.detach().cpu().numpy().reshape(-1, 3)
        col = (np.asarray(img).reshape(-1, 3) * 255).astype(np.uint8)
        m = msk.detach().cpu().numpy().reshape(-1)
        all_pts.append(pts[m])
        all_col.append(col[m])

    pts_np = np.concatenate(all_pts)
    col_np = np.concatenate(all_col)
    print(f"      Total points: {len(pts_np):,}")

    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    verts = np.empty(len(pts_np), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    verts["x"], verts["y"], verts["z"] = pts_np[:, 0], pts_np[:, 1], pts_np[:, 2]
    verts["red"], verts["green"], verts["blue"] = col_np[:, 0], col_np[:, 1], col_np[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")]).write(args.out)

    print(f"\nSaved: {args.out}")
    print("=== RECONSTRUCTION COMPLETE ===")


if __name__ == "__main__":
    main()
