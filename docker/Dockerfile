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
        # SSH 服务端
        openssh-server \
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

# ── SSH 基础配置 ─────────────────────────────────────────────────────────────
# 允许 root 通过密钥登录（RunPod 注入公钥，不需要密码）
RUN mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

ENV UV_LINK_MODE=copy

# ══════════════════════════════════════════════════════════════════════════════
# 环境 A — openpi 推理服务器  (Python 3.11)
# ══════════════════════════════════════════════════════════════════════════════
ENV UV_PROJECT_ENVIRONMENT=/.venv

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=third_party/openpi/uv.lock,target=uv.lock \
    --mount=type=bind,source=third_party/openpi/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=third_party/openpi/packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=third_party/openpi/packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev




COPY third_party/openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
        | xargs dirname \
        | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /.venv \
        -r /tmp/requirements.txt \
        jupyterlab \
        tensorboard \
        ipykernel

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /.venv pytest

# ══════════════════════════════════════════════════════════════════════════════
# 环境 B — LIBERO benchmark 客户端  (Python 3.8)
# ══════════════════════════════════════════════════════════════════════════════
ENV LIBERO_VENV=/.venv_libero

RUN uv venv --python 3.8 $LIBERO_VENV

COPY third_party/openpi/packages/openpi-client/ /tmp/openpi-client/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python $LIBERO_VENV /tmp/openpi-client

COPY third_party/openpi/examples/libero/requirements.txt /tmp/req-libero-client.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python $LIBERO_VENV \
        -r /tmp/req-libero-client.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --index-strategy=unsafe-best-match

COPY third_party/openpi/third_party/libero/requirements.txt /tmp/req-libero-lib.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python $LIBERO_VENV \
        -r /tmp/req-libero-lib.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --index-strategy=unsafe-best-match

# ── 在 3.8 环境里也装 ipykernel，然后注册到 Jupyter ──────────────────────────
# 这样你在 Jupyter 界面里可以自由切换 3.11 和 3.8 两个 kernel
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python $LIBERO_VENV ipykernel && \
    $LIBERO_VENV/bin/python -m ipykernel install \
        --name libero_py38 \
        --display-name "Python 3.8 (LIBERO)" \
        --prefix /.venv

# ── MuJoCo headless 渲染 (EGL) ────────────────────────────────────────────────
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ── LIBERO 配置文件 ───────────────────────────────────────────────────────────
ENV LIBERO_CONFIG_PATH=/opt/libero_config
RUN mkdir -p /opt/libero_config && cat > /opt/libero_config/config.yaml << 'YAML_EOF'
benchmark_root: /app/third_party/openpi/third_party/libero/libero/libero
bddl_files: /app/third_party/openpi/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/openpi/third_party/libero/libero/libero/init_files
datasets: /app/third_party/openpi/third_party/libero/libero/datasets
assets: /app/third_party/openpi/third_party/libero/libero/libero/assets
YAML_EOF

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 22
EXPOSE 8888
EXPOSE 6006

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

## =============================================================================
## Sim2Sim_Lab — 合并镜像
## 包含: openpi 推理服务器 (Python 3.11) + LIBERO benchmark 客户端 (Python 3.8)
## Build context: Sim2Sim_Lab/  (项目根目录)
## 构建前必须确保 git submodule 已完整初始化:
##   git submodule update --init --recursive
## =============================================================================
#FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
#
#COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/
#
#WORKDIR /app
#
#
#
## ── 系统依赖 ──────────────────────────────────────────────────────────────────
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        # 通用构建工具
#        git \
#        git-lfs \
#        build-essential \
#        clang \
#        make \
#        cmake \
#        g++ \
#        curl \
#        # SSH 服务端
#        openssh-server \
#        # MuJoCo / robosuite 渲染所需的 OpenGL / EGL 库
#        libosmesa6-dev \
#        libgl1-mesa-glx \
#        libegl1 \
#        libglew-dev \
#        libglfw3-dev \
#        libgles2-mesa-dev \
#        libglib2.0-0 \
#        libsm6 \
#        libxrender1 \
#        libxext6 \
#    && rm -rf /var/lib/apt/lists/*
#
## ── SSH 基础配置 ─────────────────────────────────────────────────────────────
## 允许 root 通过密钥登录（RunPod 注入公钥，不需要密码）
#RUN mkdir /var/run/sshd && \
#    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
#    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
#    sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
#
#ENV UV_LINK_MODE=copy
#
## ══════════════════════════════════════════════════════════════════════════════
## 环境 A — openpi 推理服务器  (Python 3.11)
## ══════════════════════════════════════════════════════════════════════════════
#ENV UV_PROJECT_ENVIRONMENT=/.venv
#
#RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
#
#RUN --mount=type=cache,target=/root/.cache/uv \
#    --mount=type=bind,source=third_party/openpi/uv.lock,target=uv.lock \
#    --mount=type=bind,source=third_party/openpi/pyproject.toml,target=pyproject.toml \
#    --mount=type=bind,source=third_party/openpi/packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
#    --mount=type=bind,source=third_party/openpi/packages/openpi-client/src,target=packages/openpi-client/src \
#    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev
#
#
#
#
#COPY third_party/openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
#RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
#        | xargs dirname \
#        | xargs -I{} cp -r /tmp/transformers_replace/* {} \
#    && rm -rf /tmp/transformers_replace
#
#COPY requirements.txt /tmp/requirements.txt
#RUN --mount=type=cache,target=/root/.cache/uv \
#    uv pip install --python /.venv \
#        -r /tmp/requirements.txt \
#        jupyterlab \
#        tensorboard \
#        ipykernel
#
#RUN --mount=type=cache,target=/root/.cache/uv \
#    uv pip install --python /.venv pytest
#
## ══════════════════════════════════════════════════════════════════════════════
## 环境 B — LIBERO benchmark 客户端  (Python 3.8)
## ══════════════════════════════════════════════════════════════════════════════
#ENV LIBERO_VENV=/.venv_libero
#
## ↓↓↓ 改动：加 --seed，让 uv 把 pip 本体也装进 venv，否则 /bin/pip 不存在
#RUN uv venv --python 3.8 --seed $LIBERO_VENV
#
#COPY third_party/openpi/packages/openpi-client/ /tmp/openpi-client/
#RUN --mount=type=cache,target=/root/.cache/uv \
#    uv pip install --python $LIBERO_VENV /tmp/openpi-client
#
#COPY third_party/openpi/examples/libero/requirements.txt /tmp/req-libero-client.txt
#RUN $LIBERO_VENV/bin/pip install \
#        -r /tmp/req-libero-client.txt \
#        --extra-index-url https://download.pytorch.org/whl/cu113
#
#COPY third_party/openpi/third_party/libero/requirements.txt /tmp/req-libero-lib.txt
#RUN $LIBERO_VENV/bin/pip install \
#        -r /tmp/req-libero-lib.txt \
#        --extra-index-url https://download.pytorch.org/whl/cu113
#
## ── 在 3.8 环境里也装 ipykernel，然后注册到 Jupyter ──────────────────────────
## 这样你在 Jupyter 界面里可以自由切换 3.11 和 3.8 两个 kernel
#RUN --mount=type=cache,target=/root/.cache/uv \
#    uv pip install --python $LIBERO_VENV ipykernel && \
#    $LIBERO_VENV/bin/python -m ipykernel install \
#        --name libero_py38 \
#        --display-name "Python 3.8 (LIBERO)" \
#        --prefix /.venv
#
## ── MuJoCo headless 渲染 (EGL) ────────────────────────────────────────────────
#RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
#    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
#        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
#
## ── LIBERO 配置文件 ───────────────────────────────────────────────────────────
#ENV LIBERO_CONFIG_PATH=/opt/libero_config
#RUN mkdir -p /opt/libero_config && cat > /opt/libero_config/config.yaml << 'YAML_EOF'
#benchmark_root: /app/third_party/openpi/third_party/libero/libero/libero
#bddl_files: /app/third_party/openpi/third_party/libero/libero/libero/bddl_files
#init_states: /app/third_party/openpi/third_party/libero/libero/libero/init_files
#datasets: /app/third_party/openpi/third_party/libero/libero/datasets
#assets: /app/third_party/openpi/third_party/libero/libero/libero/assets
#YAML_EOF
#
## ── Entrypoint ────────────────────────────────────────────────────────────────
#COPY docker/entrypoint.sh /entrypoint.sh
#RUN chmod +x /entrypoint.sh
#
#EXPOSE 22
#EXPOSE 8888
#EXPOSE 6006
#
#ENTRYPOINT ["/entrypoint.sh"]
#CMD ["/bin/bash"]