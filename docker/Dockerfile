# =============================================================================
# Sim2Sim_Lab — 合并镜像
# 包含: openpi 推理服务器 (Python 3.11) + LIBERO benchmark 客户端 (Python 3.8)
# Build context: Sim2Sim_Lab/  (项目根目录)
# 构建前必须确保 git submodule 已完整初始化:
#   git submodule update --init --recursive
# =============================================================================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# ── 系统依赖 ──────────────────────────────────────────────────────────────────
# 同时满足 openpi (build tools) 和 LIBERO (OpenGL/EGL/MuJoCo) 的需求
RUN apt-get update && apt-get install -y --no-install-recommends \
        # 通用构建工具
        git \
        git-lfs \
        build-essential \
        clang \
        make \
        cmake \
        g++ \
        curl \
        # MuJoCo / robosuite 渲染所需的 OpenGL / EGL 库
        libosmesa6-dev \
        libgl1-mesa-glx \
        libegl1 \
        libglew-dev \
        libglfw3-dev \
        libgles2-mesa-dev \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy

# ══════════════════════════════════════════════════════════════════════════════
# 环境 A — openpi 推理服务器  (Python 3.11)
# ══════════════════════════════════════════════════════════════════════════════
ENV UV_PROJECT_ENVIRONMENT=/.venv

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# 使用 openpi 自己的 uv.lock 精确锁定依赖（bind mount，不进入镜像层）
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=third_party/openpi/uv.lock,target=uv.lock \
    --mount=type=bind,source=third_party/openpi/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=third_party/openpi/packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=third_party/openpi/packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# 为 PyTorch VLA 模型打 transformers patch（openpi 要求）
COPY third_party/openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
        | xargs dirname \
        | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

# 安装主项目自己额外的依赖
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /.venv -r /tmp/requirements.txt

# ══════════════════════════════════════════════════════════════════════════════
# 环境 B — LIBERO benchmark 客户端  (Python 3.8)
# ══════════════════════════════════════════════════════════════════════════════
ENV LIBERO_VENV=/.venv_libero

# uv 会自动下载 Python 3.8，无需系统预装
RUN uv venv --python 3.8 $LIBERO_VENV

# ① 安装 openpi-client（LIBERO main.py 用它作为 WebSocket 客户端）
#    整个包 COPY 进来后安装，避免运行时 PYTHONPATH 依赖
COPY third_party/openpi/packages/openpi-client/ /tmp/openpi-client/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python $LIBERO_VENV /tmp/openpi-client

# ② LIBERO 评估客户端依赖
#    包含: torch==1.11.0+cu113, robosuite, mujoco, imageio, tyro …
#    cu113 wheel 不在 PyPI，必须指定额外索引
COPY third_party/openpi/examples/libero/requirements.txt /tmp/req-libero-client.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python $LIBERO_VENV \
        -r /tmp/req-libero-client.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --index-strategy=unsafe-best-match

# ③ LIBERO 库自身的依赖（hydra, robomimic, wandb, gym …）
#    注意: 此处使用小写 libero 目录名（与 openpi 仓库约定一致）
#    若你的子模块目录名为 LIBERO（大写），请相应修改
COPY third_party/openpi/third_party/libero/requirements.txt /tmp/req-libero-lib.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python $LIBERO_VENV \
        -r /tmp/req-libero-lib.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --index-strategy=unsafe-best-match

# ── MuJoCo headless 渲染 (EGL) 的 NVIDIA 厂商配置 ────────────────────────────
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ── 预置 LIBERO 配置文件，避免首次 import 时触发交互式输入提示 ─────────────────
#    路径指向运行时挂载于 /app 的 Network Volume
#    若子模块目录名大写（LIBERO），请将下方路径中的 libero 替换为 LIBERO
ENV LIBERO_CONFIG_PATH=/opt/libero_config
RUN mkdir -p /opt/libero_config && cat > /opt/libero_config/config.yaml << 'YAML_EOF'
benchmark_root: /app/third_party/openpi/third_party/libero/libero/libero
bddl_files: /app/third_party/openpi/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/openpi/third_party/libero/libero/libero/init_files
datasets: /app/third_party/openpi/third_party/libero/libero/datasets
assets: /app/third_party/openpi/third_party/libero/libero/libero/assets
YAML_EOF

CMD ["/bin/bash"]