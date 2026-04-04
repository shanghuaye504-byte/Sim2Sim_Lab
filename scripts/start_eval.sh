#!/usr/bin/env bash
set -euo pipefail




SERVER_PORT="${SERVER_PORT:-8000}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-50}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-gs://openpi-assets/checkpoints/pi05_libero}"
VIDEO_OUT="${VIDEO_OUT:-/app/data/libero/videos}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"




# ── 新增：domain config 路径，默认为空（source domain）──────────────────────
DOMAIN_CONFIG_FILE="${DOMAIN_CONFIG_FILE:-}"




# ── domain 名称仅用于日志和视频目录区分 ──────────────────────────────────────
if [ -n "${DOMAIN_CONFIG_FILE}" ]; then
    DOMAIN_NAME="$(basename "${DOMAIN_CONFIG_FILE}" .yaml)"
else
    DOMAIN_NAME="source"
fi
VIDEO_OUT="${VIDEO_OUT}/${TASK_SUITE}/${DOMAIN_NAME}"




echo "══════════════════════════════════════════════════"
echo "  Sim2Sim_Lab — Domain Shift Evaluation"
echo "══════════════════════════════════════════════════"
echo "  Task suite    : ${TASK_SUITE}"
echo "  Domain        : ${DOMAIN_NAME}"
echo "  Config file   : ${DOMAIN_CONFIG_FILE:-（source domain，无偏移）}"
echo "  Trials / task : ${NUM_TRIALS}"
echo "  Video output  : ${VIDEO_OUT}"
echo "══════════════════════════════════════════════════"




# ── 1. 启动 policy server ────────────────────────────────────────────────────
PYTHONPATH=/app/third_party/openpi/src:/app/third_party/openpi/packages/openpi-client/src:/app/src \
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/app/.cache/openpi}" \
    /.venv/bin/python \
        /app/third_party/openpi/scripts/serve_policy.py \
        --port          "${SERVER_PORT}" \
        policy:checkpoint \
        --policy.config "${POLICY_CONFIG}" \
        --policy.dir    "${POLICY_DIR}" &
SERVER_PID=$!




ELAPSED=0
until curl -sf "http://127.0.0.1:${SERVER_PORT}/healthz" > /dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[server] Policy server 意外退出。"; exit 1
    fi
    [ "${ELAPSED}" -ge "${SERVER_START_TIMEOUT}" ] && {
        echo "[server] 超时。"; kill "${SERVER_PID}" 2>/dev/null; exit 1
    }
    sleep 5; ELAPSED=$((ELAPSED + 5))
    echo "[server] 已等待 ${ELAPSED}s ..."
done
echo "[server] 服务器就绪！"




# ── 2. 启动 LIBERO 评估 ──────────────────────────────────────────────────────
mkdir -p "${VIDEO_OUT}"
LIBERO_EVAL_EXIT=0




MUJOCO_GL="${MUJOCO_GL}" \
MUJOCO_EGL_DEVICE_ID=0 \
PYOPENGL_PLATFORM=egl \
LIBERO_CONFIG_PATH=/opt/libero_config \
PYTHONPATH=/app/third_party/openpi/third_party/libero \
DOMAIN_CONFIG_FILE="${DOMAIN_CONFIG_FILE}" \
    /.venv_libero/bin/python \
        /app/eval/domain_eval.py \
        --args.host 127.0.0.1 \
        --args.port "${SERVER_PORT}" \
        --args.task-suite-name "${TASK_SUITE}" \
        --args.num-trials-per-task "${NUM_TRIALS}" \
        --args.video-out-path "${VIDEO_OUT}" \
    || LIBERO_EVAL_EXIT=$?




# ── 3. 收尾 ──────────────────────────────────────────────────────────────────
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true




echo "══════════════════════════════════════════════════"
[ "${LIBERO_EVAL_EXIT}" -eq 0 ] \
    && echo "  完成 ✓  domain=${DOMAIN_NAME}  videos=${VIDEO_OUT}" \
    || echo "  失败 ✗  exit code=${LIBERO_EVAL_EXIT}"
echo "══════════════════════════════════════════════════"
exit "${LIBERO_EVAL_EXIT}"