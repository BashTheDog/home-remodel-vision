"""
3D Gaussian Splatting trainer using gsplat 1.5.x
Input: COLMAP dense output (binary format)
Output: trained .ply

Usage:
    python3 train_3dgs.py --colmap-dir /path/to/colmap/dense \
                          --output-dir /path/to/output \
                          --name <room_name>
"""
import argparse, os, sys, struct, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

os.environ["TORCH_EXTENSIONS_DIR"] = "/project/tools/torch_ext"

from gsplat import rasterization
from gsplat import DefaultStrategy

# ── config (overridable via CLI) ──────────────────────────────────────────────
COLMAP_DIR   = "/project/outputs/bedroom1_colmap/dense"
OUTPUT_DIR   = "/project/outputs/bedroom1_3dgs"
N_ITERS      = 7000
LR_MEANS     = 1.6e-4
LR_OPACITIES = 5e-2
LR_SCALES    = 5e-3
LR_QUATS     = 1e-3
LR_COLORS    = 2.5e-3
INIT_OPACITY = 0.1
SSIM_LAMBDA  = 0.2
SH_DEGREE    = 3
MAX_DIM      = 1024        # downsample long side of images to this
DEVICE       = "cuda"
# ──────────────────────────────────────────────────────────────────────────────


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id  = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            w, h    = struct.unpack("<2Q", f.read(16))
            # SIMPLE_RADIAL (2): f, cx, cy, k
            if model_id == 2:
                f_px, cx, cy, k = struct.unpack("<4d", f.read(32))
                cameras[cam_id] = dict(w=w, h=h, fx=f_px, fy=f_px, cx=cx, cy=cy)
            # PINHOLE (1): fx, fy, cx, cy
            elif model_id == 1:
                fx, fy, cx, cy = struct.unpack("<4d", f.read(32))
                cameras[cam_id] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
            else:
                # fallback: skip params
                n_params = {0:3,3:5,4:8,5:5,6:8}.get(model_id, 4)
                f.read(n_params * 8)
                cameras[cam_id] = dict(w=w, h=h, fx=500., fy=500., cx=w/2., cy=h/2.)
    return cameras


def quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ], dtype=np.float32)


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b"".join(iter(lambda: f.read(1), b"\x00")).decode()
            n_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_pts2d * 24)
            images[img_id] = dict(name=name,
                                  R=quat_to_rot(qw, qx, qy, qz),
                                  t=np.array([tx, ty, tz], dtype=np.float32),
                                  cam_id=cam_id)
    return images


def read_points3d_bin(path):
    pts, colors = [], []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)                           # point3d_id
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            f.read(8)                           # error
            n_track = struct.unpack("<Q", f.read(8))[0]
            f.read(n_track * 8)
            pts.append([x, y, z])
            colors.append([r/255., g/255., b/255.])
    return np.array(pts, np.float32), np.array(colors, np.float32)


def ssim_loss(pred, gt):
    """pred, gt: (B, C, H, W)"""
    C1, C2 = 0.01**2, 0.03**2
    mu_p = F.avg_pool2d(pred, 11, 1, 5)
    mu_g = F.avg_pool2d(gt,   11, 1, 5)
    s_p  = F.avg_pool2d(pred**2, 11, 1, 5) - mu_p**2
    s_g  = F.avg_pool2d(gt**2,   11, 1, 5) - mu_g**2
    s_pg = F.avg_pool2d(pred*gt, 11, 1, 5) - mu_p*mu_g
    num  = (2*mu_p*mu_g + C1) * (2*s_pg + C2)
    den  = (mu_p**2 + mu_g**2 + C1) * (s_p + s_g + C2)
    return 1.0 - (num / den).mean()


def load_dataset(colmap_dir):
    sp = Path(colmap_dir) / "sparse"
    cam_path = sp / "cameras.bin" if sp.exists() else Path(colmap_dir) / "cameras.bin"
    img_path = sp / "images.bin"  if sp.exists() else Path(colmap_dir) / "images.bin"

    cameras  = read_cameras_bin(str(cam_path))
    img_meta = read_images_bin(str(img_path))
    img_dir  = Path(colmap_dir) / "images"

    frames = []
    for meta in img_meta.values():
        p = img_dir / meta["name"]
        if not p.exists():
            continue
        cam = cameras[meta["cam_id"]]
        frames.append(dict(path=str(p), R=meta["R"], t=meta["t"],
                           w=cam["w"], h=cam["h"],
                           fx=cam["fx"], fy=cam["fy"],
                           cx=cam["cx"], cy=cam["cy"]))
    print(f"Loaded {len(frames)} training views")
    return frames


def main():
    global COLMAP_DIR, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap-dir", default=COLMAP_DIR,
                    help="COLMAP dense workspace (contains sparse/ and images/)")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for .ply")
    ap.add_argument("--name", default=None,
                    help="Room name for output file (default: basename of output-dir minus '_3dgs')")
    ap.add_argument("--n-iters", type=int, default=N_ITERS)
    args = ap.parse_args()

    COLMAP_DIR = args.colmap_dir
    OUTPUT_DIR = args.output_dir
    room_name  = args.name or os.path.basename(OUTPUT_DIR).replace("_3dgs", "")
    n_iters    = args.n_iters

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── load scene ──────────────────────────────────────────────────────────
    frames = load_dataset(COLMAP_DIR)
    sp = Path(COLMAP_DIR) / "sparse"
    pts_path = sp / "points3D.bin" if sp.exists() else Path(COLMAP_DIR) / "points3D.bin"
    pts_xyz, pts_rgb = read_points3d_bin(str(pts_path))
    N = len(pts_xyz)
    print(f"Sparse cloud: {N} points")

    # ── init Gaussians ───────────────────────────────────────────────────────
    # Scale init: random sample → mean 3-NN distance → divide by 3.
    # (Sequential first-N samples hit clustered regions and overestimate spacing.)
    SH_C0  = 0.28209479177387814
    sh_dim = (SH_DEGREE + 1) ** 2

    rng = np.random.default_rng(42)
    idx = rng.choice(N, min(3000, N), replace=False)
    sub_np = torch.tensor(pts_xyz[idx], device=DEVICE)
    with torch.no_grad():
        d2 = torch.cdist(sub_np, sub_np)
        d2.fill_diagonal_(1e9)
        k3, _ = d2.topk(3, largest=False, dim=1)
        mean_3nn = k3.mean().item()
    init_log_scale = math.log(max(mean_3nn / 3.0, 1e-4))
    print(f"mean 3-NN dist={mean_3nn:.4f}m  →  init_scale={math.exp(init_log_scale):.4f}m")

    # SH init: invert DC rendering equation so first render matches point-cloud colours.
    # rendered_dc = sh0 * SH_C0 + 0.5  →  sh0 = (rgb - 0.5) / SH_C0
    sh0_init = (torch.tensor(pts_rgb, device=DEVICE) - 0.5) / SH_C0

    params = nn.ParameterDict({
        "means":     nn.Parameter(torch.tensor(pts_xyz, device=DEVICE)),
        "quats":     nn.Parameter(torch.cat([torch.ones(N,1), torch.zeros(N,3)],
                                            dim=-1).to(DEVICE)),
        "scales":    nn.Parameter(torch.full((N,3), init_log_scale, device=DEVICE)),
        "opacities": nn.Parameter(torch.full((N,),
                                  math.log(INIT_OPACITY/(1-INIT_OPACITY)),
                                  device=DEVICE)),
        "sh0":       nn.Parameter(sh0_init.unsqueeze(1)),          # (N,1,3)
        "shN":       nn.Parameter(torch.zeros(N, sh_dim-1, 3, device=DEVICE)),
    })

    optimizers = {
        "means":     torch.optim.Adam([params["means"]],     lr=LR_MEANS),
        "quats":     torch.optim.Adam([params["quats"]],     lr=LR_QUATS),
        "scales":    torch.optim.Adam([params["scales"]],    lr=LR_SCALES),
        "opacities": torch.optim.Adam([params["opacities"]], lr=LR_OPACITIES),
        "sh0":       torch.optim.Adam([params["sh0"]],       lr=LR_COLORS),
        "shN":       torch.optim.Adam([params["shN"]],       lr=LR_COLORS/20.),
    }

    strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state()

    # ── pre-load & downscale images ──────────────────────────────────────────
    print("Pre-loading images …")
    dataset = []
    for fr in frames:
        img = np.array(Image.open(fr["path"]).convert("RGB"), dtype=np.float32) / 255.
        h_raw, w_raw = img.shape[:2]
        scale = min(MAX_DIM / max(h_raw, w_raw), 1.0)
        if scale < 1.0:
            img = np.array(Image.fromarray((img*255).astype(np.uint8))
                           .resize((int(w_raw*scale), int(h_raw*scale)),
                                   Image.LANCZOS), dtype=np.float32) / 255.
        h, w = img.shape[:2]
        sx, sy = w / fr["w"], h / fr["h"]

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3,:3] = fr["R"]; w2c[:3,3] = fr["t"]
        viewmat = torch.from_numpy(w2c).unsqueeze(0).to(DEVICE)

        K = torch.tensor([[fr["fx"]*sx, 0, fr["cx"]*sx],
                          [0, fr["fy"]*sy, fr["cy"]*sy],
                          [0, 0, 1]], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        dataset.append(dict(
            img=torch.from_numpy(img).to(DEVICE),
            viewmat=viewmat, K=K, w=w, h=h,
        ))
    print(f"Training on {len(dataset)} views for {n_iters} iterations …")

    # ── training loop ────────────────────────────────────────────────────────
    t0 = time.time()
    for step in range(1, n_iters + 1):
        fr = dataset[step % len(dataset)]

        for opt in optimizers.values():
            opt.zero_grad()

        colors = torch.cat([params["sh0"], params["shN"]], dim=1)   # (N, sh_dim, 3)

        renders, alphas, info = rasterization(
            params["means"],
            F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            colors,
            fr["viewmat"], fr["K"],
            width=fr["w"], height=fr["h"],
            sh_degree=min(SH_DEGREE, step // 1000),   # anneal SH degree
            packed=True,
        )

        # strategy.step_pre_backward registers retain_grad on means2d
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        gt   = fr["img"].unsqueeze(0)
        pred = renders.clamp(0, 1)
        l1   = (pred - gt).abs().mean()
        ssim = ssim_loss(pred.permute(0,3,1,2), gt.permute(0,3,1,2))
        loss = (1 - SSIM_LAMBDA) * l1 + SSIM_LAMBDA * ssim

        loss.backward()

        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=True,
        )
        for opt in optimizers.values():
            opt.step()

        if step % 500 == 0 or step == 1:
            n = len(params["means"])
            print(f"[{step:5d}/{n_iters}] loss={loss.item():.4f}  "
                  f"l1={l1.item():.4f}  #gauss={n}  t={time.time()-t0:.0f}s")

    # ── export .ply ──────────────────────────────────────────────────────────
    print("Saving PLY …")
    from gsplat import export_splats
    with torch.no_grad():
        export_splats(
            means=params["means"],
            scales=torch.exp(params["scales"]),
            quats=F.normalize(params["quats"], dim=-1),
            opacities=torch.sigmoid(params["opacities"]),
            sh0=params["sh0"],
            shN=params["shN"],
            format="ply",
            save_to=f"{OUTPUT_DIR}/{room_name}.ply",
        )
    print(f"Saved → {OUTPUT_DIR}/{room_name}.ply")
    torch.save({"params": dict(params), "step": n_iters},
               f"{OUTPUT_DIR}/{room_name}_checkpoint.pt")
    print("Done.")


if __name__ == "__main__":
    main()
