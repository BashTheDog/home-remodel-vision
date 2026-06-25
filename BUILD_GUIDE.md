# Home Remodel Vision — Build & Reproduction Guide

**Target hardware:** 1× NVIDIA DGX Spark (GB10 Superchip, 128 GB unified memory, ARM64/aarch64, DGX OS = Ubuntu 24.04, CUDA 13.0 driver stack, compute capability sm_121)

**Goal:** A local spatial & generative vision engine for interior remodeling, running entirely on the Spark, exposing three capabilities:
1. 3D reconstruction & walkthrough (metric depth + pose/point-cloud)
2. Wall masking & swatch/material transfer (segmentation + depth-conditioned inpainting)
3. Object deletion & furniture injection (erasure + reference-image-guided generation)

**Final architecture:** A single NVIDIA AI Workbench project, built on NVIDIA's PyTorch CUDA 13.1 base container, with four model stacks installed: Depth-Pro, SAM2, GroundingDINO, DUSt3R — plus diffusers/transformers/opencv for the generative stages.

---

## 0. Critical lessons learned (read first — these are why the naive path fails)

These are the hard-won facts. An agent reproducing this should treat them as constraints, not discover them again.

1. **The host is ARM64, not x86.** Many Python wheels (e.g. `open3d`, `eva-decord`) have no aarch64 build on PyPI. Don't assume a package exists; have a fallback (`trimesh`/`plyfile` instead of `open3d`).

2. **Host CUDA is 13.0; stock PyTorch wheels target ≤12.x and do NOT support sm_121.** Installing `torch` from the standard or even nightly PyPI index produces either a CPU-only build or a build that throws `sm_121 not compatible`. **Do not try to make a bare venv work.** The fix is to use NVIDIA's prebuilt PyTorch container, which ships torch compiled for GB10.

3. **Use NVIDIA AI Workbench with the "PyTorch" base environment (CUDA 13.1, the `nvcr.io/nvidia/ai-workbench/pytorch:1.0.10` image).** This single decision eliminates the entire CUDA/torch/torchvision mess. Inside it, `torch.cuda.get_device_capability()` returns `(12, 1)` with no warning.

4. **NEVER reinstall or upgrade `torch` inside the container.** NVIDIA's build self-identifies as e.g. `2.11.0a0+...nv26.02`. Any `pip install` that pulls torch as a dependency will silently replace it with a broken generic build. Always use `--no-deps` when installing model repos, and verify torch's version string is unchanged afterward.

5. **`pip install -e` on the model repos triggers CUDA extension compilation that fails on sm_121.**
   - GroundingDINO's `setup.py` calls `pip install torch` internally AND compiles a CUDA op (`_C`). Fix: don't install it as a package at all — add its repo dir to `PYTHONPATH`. It auto-falls-back to a pure-Python implementation (you'll see `Failed to load custom C++ ops. Running on CPU mode Only!` — that warning is expected and fine).
   - Use `--no-build-isolation` for the repos you DO install editable (so they see the already-installed torch and skip reinstalling it).

6. **`opencv-python` and `opencv-python-headless` can both get installed; the full one wins and demands libGL/libxcb/libSM X11 libraries the container lacks.** Fix: uninstall BOTH, reinstall ONLY `opencv-python-headless`. This eliminates a whole class of `libGL.so.1 / libxcb.so.1: cannot open shared object file` errors. (Transformers/diffusers pull in full opencv as a transitive dep — watch for it.)

7. **`/project` is mounted only at RUNTIME, not during the build.** A `postBuild.bash` that writes to `/project` fails with `mkdir: cannot create directory '/project': Permission denied`. Guard postBuild to exit early if `/project` is absent, and do repo cloning either at runtime or interactively.

8. **`nvwb add package apt …` does NOT reliably update `.project/spec.yaml`'s apt list,** and the apt layer caches aggressively. Result: apt packages silently never install. Verify with `dpkg -l | grep <pkg>` inside the container. To force a rebuild past the cache, append a comment line to `postBuild.bash` (changes its hash). The reliable apt list lives in `.project/spec.yaml` under `installed_packages`.

9. **`variables.env` is sourced for apps Workbench launches (JupyterLab), but NOT for `nvwb attach` shells.** So PYTHONPATH set there works in notebooks but not in attached terminals. The robust fix is a `.pth` file in site-packages, which Python reads in every context.

10. **The `nvwb` CLI binary is named `nvwb-cli` on disk but calls itself `nvwb`.** Internal subcommands (`activate`) fail with `exec: "nvwb": executable file not found in $PATH` until you symlink it: `ln -s ~/.nvwb/bin/nvwb-cli ~/.nvwb/bin/nvwb` and add `~/.nvwb/bin` to PATH.

11. **`nvwb` command quirks:** context is `local` (the Spark sees itself as local). `deactivate` takes no argument. `open`/`status` need an active context first (`nvwb activate local`). `open` only works on projects already in the Workbench inventory — a raw `git clone` is NOT registered; use `nvwb create project` or `nvwb clone project`. The Workbench service must be running (`nvwb activate local` starts it).

12. **A hand-written `spec.yaml` will fail validation** with `missing required project mount in execution.mounts`. Don't hand-author it — let `nvwb create project` generate a valid one, then edit only the package lists.

---

## 1. Final working environment (target state to reproduce)

| Component | Value |
|---|---|
| Workbench project name | `home-remodel-vision` |
| Project path on host | `/home/pharn/nvidia-workbench/hrv` |
| Workbench context | `local` |
| Base environment | PyTorch / CUDA 13.1 (`nvcr.io/nvidia/ai-workbench/pytorch:1.0.10`) |
| torch version (in container) | `2.11.0a0+...nv26.02` (NVIDIA build, sm_121) |
| Container home | `/home/ubuntu` (NOT `/home/workbench`; user is `workbench`) |
| Runtime project mount | `/project` → `/home/pharn/nvidia-workbench/hrv` |
| Model repos location | `/project/repos/{ml-depth-pro,sam2,dust3r,GroundingDINO}` |
| GitHub remote | `https://github.com/BashTheDog/home-remodel-vision.git` |

### Models / repos
- **Depth-Pro** (Apple) — metric depth. Installed editable `--no-deps`.
- **SAM2** (Meta) — segmentation + tracking. Installed editable `--no-deps`. Needs `hydra-core`, `iopath`.
- **GroundingDINO** (IDEA) — open-vocab detection. **PYTHONPATH only, NOT pip-installed** (CUDA compile fails; pure-Python fallback used).
- **DUSt3R** (NAVER) — pose + dense point cloud. **PYTHONPATH only.** Needs `roma`.

### apt packages (in `.project/spec.yaml` installed_packages)
`curl, git, git-lfs, vim, ffmpeg, libgl1, libglvnd0, libglib2.0-0t64, libxcb1, libxrender1, libxext6, libsm6, build-essential, cmake, ninja-build`
(Note: opencv-headless still needs some of these; but the real opencv fix is the headless-only reinstall, item 6 above.)

### pip packages
Core: `transformers diffusers accelerate huggingface_hub[torch] opencv-python-headless pillow scipy matplotlib timm omegaconf tqdm imageio imageio-ffmpeg`
Geometry/seg: `trimesh plyfile scikit-image addict yapf supervision pycocotools`
Repo deps: `pillow_heif hydra-core iopath roma pyglet tensorboard`
**Excluded on purpose:** `torch`, `torchvision` (use NVIDIA's), `opencv-python` (headless only), `eva-decord` (no ARM wheel, not needed).

---

## 2. Reproduction steps (clean machine → working pipeline)

### Step 1 — NVIDIA Sync + Workbench (from a Mac/PC)
1. Install **NVIDIA Sync** on the client; add the DGX Spark (hostname + credentials).
2. From Sync, install/launch **AI Workbench**; it connects to the Spark as the `local` context.

### Step 2 — Fix the nvwb CLI PATH (on the Spark, via Sync Terminal)
```bash
ln -s ~/.nvwb/bin/nvwb-cli ~/.nvwb/bin/nvwb
export PATH="$HOME/.nvwb/bin:$PATH"
echo 'export PATH="$HOME/.nvwb/bin:$PATH"' >> ~/.bashrc
nvwb activate local      # starts the Workbench service; prompt becomes (nvwb:local)
```

### Step 3 — Create the project with the PyTorch CUDA 13.1 base
Get the base ID (it's a base64 blob; find the one named "PyTorch", CUDA 13.1):
```bash
nvwb list bases -o json   # locate Name=="PyTorch"; copy its "Id"
```
Create (this generates a VALID spec.yaml — do not hand-write one):
```bash
nvwb create project home-remodel-vision \
  --base-environment-id <PYTORCH_CUDA_13.1_BASE_ID> \
  --description "Local spatial and generative vision engine" \
  --projectPath /home/pharn/nvidia-workbench/hrv
```
If you get `workbench not available at http://localhost:10001`, run `nvwb deactivate` then `nvwb activate local` and retry.

Open it:
```bash
nvwb open /home/pharn/nvidia-workbench/hrv
```

### Step 4 — Verify the GPU base (the whole point)
```bash
nvwb start jupyterlab
nvwb attach
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# Expect: 2.11.0a0+...nv26.02  True  (12, 1)   ← no sm_121 warning
exit
```

### Step 5 — Register apt packages in spec.yaml
Edit `/home/pharn/nvidia-workbench/hrv/.project/spec.yaml`; under the `apt` package manager's `installed_packages:` list, add:
```
- ffmpeg
- libgl1
- libglvnd0
- libglib2.0-0t64
- libxcb1
- libxrender1
- libxext6
- libsm6
- build-essential
- cmake
- ninja-build
```
Force a rebuild past the cache (append a trigger comment to postBuild.bash, then build):
```bash
echo "# rebuild trigger $(date +%s)" >> /home/pharn/nvidia-workbench/hrv/postBuild.bash
nvwb build
dpkg ...   # verify inside container: nvwb attach; dpkg -l | grep libxcb1
```

### Step 6 — Register pip packages
```bash
nvwb add package pip transformers diffusers accelerate huggingface_hub \
  opencv-python-headless pillow scipy matplotlib timm omegaconf tqdm \
  imageio imageio-ffmpeg trimesh plyfile scikit-image addict yapf \
  supervision pycocotools
```
Then verify torch survived:
```bash
nvwb attach
python3 -c "import torch; print(torch.__version__)"   # must still be the nv26.02 build
```

### Step 7 — Fix opencv (eliminates libGL/libxcb import errors)
```bash
# inside container (nvwb attach)
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-python-headless
python3 -c "import cv2; print('opencv OK', cv2.__version__)"
```

### Step 8 — Clone model repos into the persistent mount
```bash
# inside container, /project is mounted at runtime
mkdir -p /project/repos
git clone https://github.com/apple/ml-depth-pro.git /project/repos/ml-depth-pro
git clone https://github.com/facebookresearch/sam2.git /project/repos/sam2
git clone --recursive https://github.com/naver/dust3r.git /project/repos/dust3r
git clone https://github.com/IDEA-Research/GroundingDINO.git /project/repos/GroundingDINO
```

### Step 9 — Install repo dependencies (one pass; torch/torchvision/opencv-python excluded)
```bash
pip install pillow_heif timm matplotlib hydra-core iopath \
  roma pyglet "huggingface-hub[torch]>=0.22" tensorboard \
  addict yapf "supervision>=0.22.0" pycocotools trimesh plyfile scikit-image
```

### Step 10 — Editable-install the two repos that tolerate it (NEVER without --no-deps)
```bash
pip install --no-deps --no-build-isolation -e /project/repos/ml-depth-pro
pip install --no-deps --no-build-isolation -e /project/repos/sam2
# DUSt3R and GroundingDINO are NOT installed — they run via PYTHONPATH (.pth below)
```

### Step 11 — Download GroundingDINO weights
```bash
mkdir -p /project/repos/GroundingDINO/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -O /project/repos/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

### Step 12 — Make DUSt3R + GroundingDINO importable everywhere (.pth, not env var)
```bash
SP=$(python3 -c "import site; print(site.getusersitepackages())")  # /home/ubuntu/.local/lib/python3.12/site-packages
printf '/project/repos/GroundingDINO\n/project/repos/dust3r\n' > "$SP/hrv_repos.pth"
```
Also set PYTHONPATH in `variables.env` for notebooks (belt-and-suspenders):
```
PYTHONPATH=/project/repos/GroundingDINO:/project/repos/dust3r
```

### Step 13 — Guard postBuild against build-time /project absence
In `/home/pharn/nvidia-workbench/hrv/postBuild.bash`, ensure it begins:
```bash
set -e
REPOS=/project/repos
if [ ! -d /project ]; then
  echo "postBuild: /project not mounted (build time) — skipping."
  exit 0
fi
mkdir -p "$REPOS"
# ... (repo clone/install logic here is for runtime reproduction)
```

### Step 14 — Verify the full pipeline
```bash
# inside container, no manual PYTHONPATH needed (.pth handles it)
python3 -c "
import torch
print('torch:', torch.__version__, '| cc:', torch.cuda.get_device_capability(0))
import transformers, diffusers, cv2, trimesh
from plyfile import PlyData
print('core libs OK')
import depth_pro; print('depth_pro OK')
from sam2.build_sam import build_sam2; print('sam2 OK')
import groundingdino
from groundingdino.util.inference import load_model
print('groundingdino OK')   # 'Failed to load custom C++ ops ... CPU mode' warning is EXPECTED
from dust3r.inference import inference; print('dust3r OK')
print('=== ALL IMPORTS PASS ===')
"
```
Expected harmless warnings: `timm.models.layers deprecated`, `Failed to load custom C++ ops. Running on CPU mode Only!` (GroundingDINO pure-Python fallback), `invalid escape sequence`.

### Step 15 — Commit the recipe (NOT the repos)
```bash
cd /home/pharn/nvidia-workbench/hrv
printf '\n# Pipeline model repos (cloned at runtime, not version-controlled)\nrepos/\n' >> .gitignore
git add -A
git commit -m "Working pipeline: CUDA 13.1 base, all 4 model repos importing on GB10"
git remote add origin https://github.com/BashTheDog/home-remodel-vision.git  # if not set
git push -u origin main --force
```

---

## 3. Known remaining work (not yet done)

1. **Full from-scratch reproducibility is incomplete.** The pip installs (Steps 6/9/10) and the `.pth` file (Step 12) were applied interactively into the running container. They persist while the container exists, but a `--full-build` on a brand-new machine won't recreate them automatically, because postBuild skips at build time (Step 13 guard) and not every interactive `pip install` is captured in `requirements.txt`. **To harden:** add a runtime startup hook (or a first-run script) that performs Steps 8–12 idempotently, and/or pin every package in `requirements.txt`.

2. **No inference has been run yet** — only imports verified. Next: drop room photos into `/project/data` and run a depth-estimation or wall-segmentation pass.

3. **Pipeline application code not yet written** — the depth/segment/diffusion modules and any API/UI layer. Earlier multi-container compose design was abandoned in favor of this single-container approach (the CUDA mismatch it worked around no longer exists).

4. **numpy version note:** depth_pro declares `numpy<2` but runs fine on the container's numpy 2.x. If a real runtime conflict appears, revisit.

---

## 4. Quick reference — daily startup

```bash
# On the Spark (or via Sync Terminal)
export PATH="$HOME/.nvwb/bin:$PATH"
nvwb activate local
nvwb open /home/pharn/nvidia-workbench/hrv
nvwb start jupyterlab          # reach it from the Mac via NVIDIA Sync, not raw localhost
nvwb attach                    # shell inside the container
```
