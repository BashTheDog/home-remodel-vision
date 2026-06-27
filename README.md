# Home Remodel Vision

A local vision pipeline for interior remodeling, running on a DGX Spark (GB10, ARM64, CUDA 13.1).
Built stage by stage inside NVIDIA AI Workbench.

---

## Stages

| # | Stage | Script | Status |
|---|---|---|---|
| 1 | Vertical panorama stitch | `stitch_pano.sh` | ✅ Done |
| 2 | Wall masking | — | Planned |
| 3 | Material / swatch transfer | — | Planned |
| 4 | Object removal + furniture injection | — | Planned |

---

## Stage 1 — Vertical Panorama Stitch

Stitch an upper and lower photo of the same scene into one tall image.

### Run it

```bash
bash /project/stitch_pano.sh <upper_image> <lower_image> <output_name>
```

| Arg | Example | Notes |
|---|---|---|
| `upper_image` | `data/pano/upper.jpg` | path to the upper photo (abs or relative to `/project`) |
| `lower_image` | `data/pano/lower.jpg` | path to the lower photo |
| `output_name` | `living_room` | basename only — output lands in `outputs/pano/<name>.jpg` |

### Example

```bash
bash /project/stitch_pano.sh data/pano/IMG_2452.JPG data/pano/IMG_2453.JPG room1
```

### How it works

1. Tries `cv2.Stitcher` (SCANS / flat projection) first.
2. Falls back to SIFT feature detection (at ¼ scale for speed) → FLANN matching → RANSAC homography → feathered vertical blend.
3. Output is a single JPEG in `outputs/pano/`.

### View in browser

After stitching, start the viewer:

```bash
python3 /project/tools/pano_viewer/serve.py outputs/pano/room1.jpg
```

It prints the exact SSH tunnel command with the container's IP, e.g.:

```
Mac tunnel  →  ssh -L 8081:172.18.0.3:8081 spark-pharn.local
Then open   →  http://localhost:8081/
```

The viewer supports scroll-to-zoom, drag-to-pan, and double-click-to-fit.

### Files

```
stitch_pano.sh                    ← shell wrapper
stitch_pano.py                    ← stitching logic
tools/pano_viewer/
    serve.py                      ← HTTP server (port 8081, 0.0.0.0)
    index.html                    ← pan/zoom viewer
outputs/pano/                     ← stitched results (gitignored)
data/pano/                        ← input photos (gitignored)
```

---

## Environment

| Component | Value |
|---|---|
| Hardware | DGX Spark GB10, 128 GB unified memory, ARM64 |
| OS | DGX OS (Ubuntu 24.04) |
| Container | NVIDIA AI Workbench — PyTorch / CUDA 13.1 |
| torch | `2.11.0a0+...nv26.02` (NVIDIA build, sm_121) |
| Key constraint | Never reinstall or upgrade torch; use `--no-deps` for any pip install |

See `BUILD_GUIDE.md` for the full reproduction guide, including all hard-won environment lessons.

---

## SSH tunnel note

Scripts print the tunnel command with the **container's IP** (not `localhost`), because
the server runs inside the Workbench container on a Docker bridge network:

```bash
ssh -L 8081:<container-ip>:8081 <spark-hostname>
```

If the container restarts and gets a new IP, run `hostname -I | awk '{print $1}'`
inside the container to get the current one.

---

## Archived — 3DGS pipeline

The earlier 3D Gaussian Splat reconstruction pipeline (COLMAP + gsplat) is preserved in
`archive_3d/`. It is no longer the active direction.
