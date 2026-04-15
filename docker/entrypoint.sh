#!/usr/bin/env bash
set -euo pipefail

APP_OVERRIDE="${APP_OVERRIDE:-/workspace/Sim2Sim_Lab}"

# ── Step 1: Symlink override /app ─────────────────────────────────────────────
if [ "${APP_OVERRIDE}" != "/app" ]; then
    if [ -d "${APP_OVERRIDE}" ] || [ -L "${APP_OVERRIDE}" ]; then
        # Only redirect if target exists, to avoid symlinking a non-existent path which causes 404 errors
        echo "[entrypoint] Redirecting /app -> ${APP_OVERRIDE}"

        # If /app is already a symlink pointing to the correct target, skip (idempotent)
        if [ "$(readlink /app 2>/dev/null)" = "${APP_OVERRIDE}" ]; then
            echo "[entrypoint] /app already points to ${APP_OVERRIDE}, skipping."
        else
            rm -rf /app
            ln -s "${APP_OVERRIDE}" /app
            echo "[entrypoint] Symlink created: /app -> $(readlink /app)"
        fi
    else
        echo "[entrypoint] Warning: ${APP_OVERRIDE} does not exist, skipping redirect."
        echo "[entrypoint] Please ensure RunPod Network Volume is correctly mounted at /workspace."
    fi
fi

# ── Step 2: Explicitly set PYTHONPATH ─────────────────────────────────────────
export PYTHONPATH=\
/app/third_party/openpi/src:\
/app/third_party/openpi/third_party/libero:\
/app/third_party/openpi/packages/openpi-client/src:\
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

# ── 5. WandB does not require a separate service ─────────────────────────────
# Just set the WANDB_API_KEY env variable in the RunPod template; the wandb library reads it automatically

echo "================================================"
echo " All services started, entering CMD..."
echo "================================================"

exec "$@"
