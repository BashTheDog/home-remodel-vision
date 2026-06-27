#!/bin/bash
# Executed at the end of every container build.
# Clones pipeline repos into the PERSISTENT /project/repos mount (survives rebuilds)
# and installs all dependencies. Idempotent: re-clones only if a repo is absent.
set -e

REPOS=/project/repos

# /project is only mounted at RUNTIME, not during build.
# If it's absent (build time), skip — repos already persist in the host mount.
if [ ! -d /project ]; then
    echo "postBuild: /project not mounted (build time) — skipping repo setup."
    exit 0
fi
mkdir -p "$REPOS"

# ── Model repos (clone only if missing — /project persists on host) ──
[ -d "$REPOS/ml-depth-pro" ] || git clone https://github.com/apple/ml-depth-pro.git "$REPOS/ml-depth-pro"
[ -d "$REPOS/sam2" ]         || git clone https://github.com/facebookresearch/sam2.git "$REPOS/sam2"
[ -d "$REPOS/dust3r" ]       || git clone --recursive https://github.com/naver/dust3r.git "$REPOS/dust3r"
[ -d "$REPOS/GroundingDINO" ] || git clone https://github.com/IDEA-Research/GroundingDINO.git "$REPOS/GroundingDINO"

# ── Editable installs (--no-deps protects NVIDIA torch; --no-build-isolation avoids torch reinstall) ──
pip install --no-deps --no-build-isolation -e "$REPOS/ml-depth-pro"
pip install --no-deps --no-build-isolation -e "$REPOS/sam2"
# DUSt3R and GroundingDINO run via PYTHONPATH (set in variables.env), no install needed

# ── All pipeline dependencies in one pass (torch/torchvision/opencv-python excluded on purpose) ──
pip install \
    pillow_heif timm matplotlib \
    hydra-core iopath \
    roma pyglet "huggingface-hub[torch]>=0.22" tensorboard \
    addict yapf "supervision>=0.22.0" pycocotools \
    trimesh plyfile scikit-image \
    gsplat

# ── GroundingDINO weights ──
mkdir -p "$REPOS/GroundingDINO/weights"
[ -f "$REPOS/GroundingDINO/weights/groundingdino_swint_ogc.pth" ] || \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
        -O "$REPOS/GroundingDINO/weights/groundingdino_swint_ogc.pth"

echo "Pipeline repos and dependencies installed."

# ── Claude Code CLI ──
# Install user-scope (no sudo); ~/.nvm and ~/.local/bin are user-writable.
# The install script uses curl internally, which is present in the container.
if ! command -v claude &>/dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash
    echo "Claude Code installed."
else
    echo "Claude Code already present: $(claude --version 2>/dev/null || true)"
fi
