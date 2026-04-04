#!/usr/bin/env bash
set -euo pipefail

APP_OVERRIDE="${APP_OVERRIDE:-/workspace/Sim2Sim_Lab}"

# ── Step 1: 软链接覆盖 /app ───────────────────────────────────────────────────
if [ "${APP_OVERRIDE}" != "/app" ]; then
    if [ -d "${APP_OVERRIDE}" ] || [ -L "${APP_OVERRIDE}" ]; then
        # 目标存在才做重定向，避免把不存在的路径软链接进来导致后续全部 404
        echo "[entrypoint] 重定向 /app -> ${APP_OVERRIDE}"

        # 如果 /app 已经是指向正确目标的软链接，跳过（幂等）
        if [ "$(readlink /app 2>/dev/null)" = "${APP_OVERRIDE}" ]; then
            echo "[entrypoint] /app 已正确指向 ${APP_OVERRIDE}，跳过。"
        else
            rm -rf /app
            ln -s "${APP_OVERRIDE}" /app
            echo "[entrypoint] 软链接创建成功: /app -> $(readlink /app)"
        fi
    else
        echo "[entrypoint] 警告：${APP_OVERRIDE} 不存在，跳过重定向。"
        echo "[entrypoint] 请确认 RunPod Network Volume 已正确挂载到 /workspace。"
    fi
fi

# ── Step 2: 显式设置 PYTHONPATH ──────────────────────────────────────────────
export PYTHONPATH=\
/app/third_party/openpi/src:\
/app/third_party/openpi/third_party/libero:\
${PYTHONPATH:-}

echo "[entrypoint] PYTHONPATH=${PYTHONPATH}"


# ── 3. JupyterLab ────────────────────────────────────────────────────────────
JUPYTER_ROOT=${APP_OVERRIDE:-/app}
/.venv/bin/jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --notebook-dir="${JUPYTER_ROOT}" \
    > /tmp/jupyter.log 2>&1 &
echo "✓ JupyterLab started on :8888  (log: /tmp/jupyter.log)"

# ── 4. TensorBoard ───────────────────────────────────────────────────────────
TB_LOGDIR=${APP_OVERRIDE:-/app}/logs
mkdir -p "${TB_LOGDIR}"
/.venv/bin/tensorboard \
    --logdir="${TB_LOGDIR}" \
    --host=0.0.0.0 \
    --port=6006 \
    > /tmp/tensorboard.log 2>&1 &
echo "✓ TensorBoard started on :6006  (log: /tmp/tensorboard.log)"

# ── 5. WandB 无需启动服务 ────────────────────────────────────────────────────
# 只需在 RunPod 模板里设置环境变量 WANDB_API_KEY 即可，wandb 库会自动读取

echo "================================================"
echo " 所有服务已启动，进入 CMD..."
echo "================================================"

exec "$@"