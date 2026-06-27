# Home Remodel Vision — Pipeline README

Reconstruct any room from a walk-through video clip into a 3D Gaussian Splat you can explore in a browser.

---

## Quick start

```bash
# 1. Reconstruct a room
bash /project/process_room.sh <start_time> <end_time> <room_name> [fps]

# 2. View it
bash /project/view_room.sh <room_name>
```

---

## process_room.sh

**Extracts frames → COLMAP SfM → 3DGS training → .ply splat**

```
bash /project/process_room.sh <start_time> <end_time> <room_name> [fps]
```

| Arg | Example | Notes |
|---|---|---|
| `start_time` | `00:01:30` or `90` | ffmpeg time — seconds or HH:MM:SS |
| `end_time`   | `00:02:10` or `130` | pick a 30–90 s walk-through |
| `room_name`  | `kitchen` | slug for output directory names |
| `fps`        | `2` *(default)* | frames per second to extract; 2 fps → ~60 frames for 30 s |

### Example

```bash
bash /project/process_room.sh 00:00:30 00:01:10 kitchen
```

### What it does

1. **Frame extraction** — ffmpeg trims the clip, rotates 90° clockwise (`transpose=1`), scales width to 1280 px, saves JPEG frames at the requested fps.
2. **COLMAP feature extraction** — SIFT features on GPU.
3. **COLMAP sequential matching** — matches consecutive frames (fast for video).
4. **COLMAP mapper** — incremental SfM, produces camera poses + sparse point cloud.
5. **3DGS training** — 7000 iterations, gsplat 1.5.x. Takes ~15–30 min on the GB10.

### Outputs

```
/project/data/frames/<room_name>/       ← JPEG frames
/project/outputs/<room_name>_colmap/    ← COLMAP workspace + database
/project/outputs/<room_name>_3dgs/
    <room_name>.ply              ← trained splat (load into viewer)
    <room_name>_checkpoint.pt    ← gsplat training checkpoint
```

### Gotchas

- **Minimum ~30 s clip, 2 fps** — COLMAP needs ≥20 registered frames for a good splat; too few and training degrades.
- **Slow or static shots fail COLMAP** — the sequential matcher needs feature motion between frames. Pick a walking shot, not a pan from a tripod.
- **Video source** — currently hardcoded to `/project/data/IMG_2445.MOV`. To use a different file, edit the `VIDEO=` line in `process_room.sh`.
- **Room name must be a slug** (letters, digits, underscore) — it becomes a directory name.
- **Re-running** overwrites existing frames and COLMAP workspace; the 3DGS output dir is preserved if already present.

---

## view_room.sh

**Starts a local HTTP server and tells you the SSH tunnel command.**

```
bash /project/view_room.sh <room_name> [port]
```

| Arg | Default | Notes |
|---|---|---|
| `room_name` | `bedroom1` | must match what you passed to `process_room.sh` |
| `port` | `8080` | TCP port inside the container |

### Example

```bash
bash /project/view_room.sh kitchen
```

### How to open it on your Mac

1. Run `view_room.sh` — it prints the SSH tunnel command.
2. In a **new Mac terminal**, run:
   ```bash
   ssh -L 8080:localhost:8080 <spark-hostname>
   ```
   (Replace `<spark-hostname>` with the DGX Spark host or IP.)
3. Open **http://localhost:8080/** in Chrome or Safari.
4. The splat loads and you can orbit/zoom/pan.

### Controls

| Action | Input |
|---|---|
| Orbit | Left-drag |
| Zoom | Scroll wheel |
| Pan | Right-drag |

### Gotchas

- **One room at a time** — the server symlinks the chosen room's `.ply` to `scene.ply`; restart the server to switch rooms.
- **Port 8080 in use** — run `pkill -f serve.py` then retry, or pass a different port: `bash view_room.sh kitchen 8081` and adjust the SSH tunnel accordingly.
- **Large PLY files take 10–30 s to load** in the browser — the progress bar shows download progress.
- **SharedArrayBuffer requirement** — the viewer uses `Cross-Origin-Embedder-Policy: require-corp`. Some browsers block this for localhost; Chrome works reliably.

---

## How the viewer works

```
/project/tools/viewer/
    serve.py                     ← Python HTTP server (no npm/node required)
    index.html                   ← viewer UI
    three.module.js              ← three.js r176 extras layer
    three.core.js                ← three.js r176 core (fetched from unpkg at setup)
    gaussian-splats-3d.module.js ← GaussianSplats3D splat renderer
    OrbitControls.js             ← orbit/pan/zoom controls
    scene.ply → (symlink)        ← points to the active room's .ply
```

---

## Re-running after a container rebuild

The container is ephemeral; `/project` is the persistent host mount. After a rebuild:

1. **Python deps** (gsplat, torch, etc.) are re-installed by `postBuild.bash` at container start.
2. **COLMAP** lives in `/project/tools/envs/colmap/` — persists.
3. **Trained splats** live in `/project/outputs/` — persists.
4. **Source video** is in `/project/data/IMG_2445.MOV` — persists.
5. **Claude Code** is re-installed by `postBuild.bash` (curl install, user-scope, no sudo).

---

## Changing training length

Default is 7000 iterations (~15–30 min). Pass `--n-iters` directly to train faster:

```bash
python3 /project/train_3dgs.py \
    --colmap-dir /project/outputs/kitchen_colmap/dense \
    --output-dir /project/outputs/kitchen_3dgs \
    --name kitchen \
    --n-iters 3000
```

---

## Full example end-to-end

```bash
# Inside the Workbench container (nvwb attach):

# Reconstruct a kitchen segment (3:00–3:40 in the video, 2 fps = ~80 frames)
bash /project/process_room.sh 00:03:00 00:03:40 kitchen

# View it
bash /project/view_room.sh kitchen

# On Mac (separate terminal):
ssh -L 8080:localhost:8080 <spark-hostname>
# open http://localhost:8080/
```
