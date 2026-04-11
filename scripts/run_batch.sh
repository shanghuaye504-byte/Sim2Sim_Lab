#!/usr/bin/env bash
set -euo pipefail

##############################################################################
#  Sim2Sim_Lab — 批量实验脚本  run_batch.sh
#
#  用法:
#    bash /app/eval/run_batch.sh
#
#  可通过环境变量覆盖默认值，例如:
#    NUM_TRIALS=10 bash /app/eval/run_batch.sh
##############################################################################

# ═══════════════════════════  配置区域  ═══════════════════════════════════════



# 模型检查点，格式: "policy_config|checkpoint_dir"
# 如需多个 checkpoint，逐行添加即可
CHECKPOINTS=(
    "pi05_libero|gs://openpi-assets/checkpoints/pi05_libero"
    # "another_policy|/app/data/base_checkpoints/another_policy"
)

# 任务套件
TASK_SUITES=(
    "libero_spatial"
    "libero_object"
    "libero_goal"
    "libero_10"
)

# Domain 配置目录
DOMAIN_CONFIG_DIR="/app/eval/domain_configs"

# 每个 task 的 trial 数
NUM_TRIALS="${NUM_TRIALS:-1}"

# 服务器端口
SERVER_PORT="${SERVER_PORT:-8000}"

# 输出根目录
RESULTS_ROOT="${RESULTS_ROOT:-/app/data/libero}"
VIDEO_ROOT="${RESULTS_ROOT}/videos"
LOG_ROOT="${RESULTS_ROOT}/logs"

# 服务器启动超时（秒）
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"

# MuJoCo 渲染
MUJOCO_GL="${MUJOCO_GL:-egl}"

# ═══════════════════════════  函数  ═══════════════════════════════════════════

SERVER_PID=""

start_server() {
    local policy_config="$1"
    local policy_dir="$2"

    echo "[server] 启动 policy server: config=${policy_config}, dir=${policy_dir}"

    PYTHONPATH=/app/third_party/openpi/src:/app/third_party/openpi/packages/openpi-client/src:/app/src \
    OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/app/.cache/openpi}" \
        /.venv/bin/python \
            /app/third_party/openpi/scripts/serve_policy.py \
            --port "${SERVER_PORT}" \
            policy:checkpoint \
            --policy.config "${policy_config}" \
            --policy.dir "${policy_dir}" &
    SERVER_PID=$!

    local elapsed=0
    until curl -sf "http://127.0.0.1:${SERVER_PORT}/healthz" > /dev/null 2>&1; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[server] ❌ Policy server 意外退出"; return 1
        fi
        if [ "${elapsed}" -ge "${SERVER_START_TIMEOUT}" ]; then
            echo "[server] ❌ 启动超时 (${SERVER_START_TIMEOUT}s)"
            kill "${SERVER_PID}" 2>/dev/null; return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        [ $((elapsed % 30)) -eq 0 ] && echo "[server] 已等待 ${elapsed}s ..."
    done
    echo "[server] ✅ 服务器就绪 (PID=${SERVER_PID})"
}

stop_server() {
    if [ -n "${SERVER_PID}" ]; then
        echo "[server] 停止 policy server (PID=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}

run_single_eval() {
    local task_suite="$1"
    local domain_config_file="$2"  # 空字符串表示 source domain
    local domain_name="$3"
    local model_name="$4"

    local log_dir="${LOG_ROOT}/${model_name}/${task_suite}/${domain_name}"
    local video_dir="${VIDEO_ROOT}/${model_name}/${task_suite}/${domain_name}"
    mkdir -p "${log_dir}" "${video_dir}"

    MUJOCO_GL="${MUJOCO_GL}" \
    MUJOCO_EGL_DEVICE_ID=0 \
    PYOPENGL_PLATFORM=egl \
    LIBERO_CONFIG_PATH=/opt/libero_config \
    PYTHONPATH=/app/third_party/openpi/third_party/libero \
    DOMAIN_CONFIG_FILE="${domain_config_file}" \
        /.venv_libero/bin/python \
            /app/eval/domain_eval.py \
            --args.host 127.0.0.1 \
            --args.port "${SERVER_PORT}" \
            --args.task-suite-name "${task_suite}" \
            --args.num-trials-per-task "${NUM_TRIALS}" \
            --args.video-out-path "${video_dir}" \
            --args.log-dir "${log_dir}"
}

# ═══════════════════════════  构建 Domain 列表  ═══════════════════════════════

declare -a DOMAIN_ENTRIES=("source|")   # source domain: 无配置文件

if [ -d "${DOMAIN_CONFIG_DIR}" ]; then
    for f in "${DOMAIN_CONFIG_DIR}"/*.yaml; do
        [ -f "$f" ] || continue
        name="$(basename "$f" .yaml)"
        DOMAIN_ENTRIES+=("${name}|${f}")
    done
fi

# ═══════════════════════════  计算实验总数  ═══════════════════════════════════

TOTAL_PLANNED=$(( ${#CHECKPOINTS[@]} * ${#TASK_SUITES[@]} * ${#DOMAIN_ENTRIES[@]} ))

# ═══════════════════════════  主循环  ═════════════════════════════════════════

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_ROOT}/batch_summary_${TIMESTAMP}.log"
mkdir -p "${RESULTS_ROOT}"

# 脚本退出时确保关闭 server
trap stop_server EXIT

{
echo "══════════════════════════════════════════════════════════"
echo "  Sim2Sim_Lab — 批量实验"
echo "  开始时间      : $(date)"
echo "  检查点数      : ${#CHECKPOINTS[@]}"
echo "  任务套件      : ${TASK_SUITES[*]}"
echo "  Domain 数     : ${#DOMAIN_ENTRIES[@]} (含 source)"
echo "  每 task trial : ${NUM_TRIALS}"
echo "  计划实验总数  : ${TOTAL_PLANNED}"
echo "══════════════════════════════════════════════════════════"
echo ""
} | tee "${SUMMARY}"

for ckpt_entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r POLICY_CONFIG POLICY_DIR <<< "${ckpt_entry}"
    MODEL_NAME="$(basename "${POLICY_DIR}")"

    echo "" | tee -a "${SUMMARY}"
    echo "▶ 模型: ${MODEL_NAME} (config=${POLICY_CONFIG})" | tee -a "${SUMMARY}"
    echo "  路径: ${POLICY_DIR}" | tee -a "${SUMMARY}"
    echo "" | tee -a "${SUMMARY}"

    # 每个 checkpoint 只启动一次 server
    if ! start_server "${POLICY_CONFIG}" "${POLICY_DIR}"; then
        echo "  [FATAL] 无法启动 server，跳过此 checkpoint" | tee -a "${SUMMARY}"
        # 将该 checkpoint 下所有实验标记为 FAILED
        local_fail_count=$(( ${#TASK_SUITES[@]} * ${#DOMAIN_ENTRIES[@]} ))
        TOTAL=$((TOTAL + local_fail_count))
        FAILED=$((FAILED + local_fail_count))
        continue
    fi

    for task_suite in "${TASK_SUITES[@]}"; do
        for domain_entry in "${DOMAIN_ENTRIES[@]}"; do
            IFS='|' read -r DOMAIN_NAME DOMAIN_CONFIG_FILE <<< "${domain_entry}"
            TOTAL=$((TOTAL + 1))

            # ── 断点续跑：JSON 已存在则跳过 ──
            json_path="${LOG_ROOT}/${MODEL_NAME}/${task_suite}/${DOMAIN_NAME}/eval_results.json"
            if [ -f "${json_path}" ]; then
                echo "  [SKIP] ${task_suite} / ${DOMAIN_NAME}" | tee -a "${SUMMARY}"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo "  [RUN]  ${task_suite} / ${DOMAIN_NAME}  (${TOTAL}/${TOTAL_PLANNED})" | tee -a "${SUMMARY}"

            # 确保日志目录存在（tee 需要）
            local_log_dir="${LOG_ROOT}/${MODEL_NAME}/${task_suite}/${DOMAIN_NAME}"
            mkdir -p "${local_log_dir}"

            if run_single_eval \
                "${task_suite}" \
                "${DOMAIN_CONFIG_FILE}" \
                "${DOMAIN_NAME}" \
                "${MODEL_NAME}" \
                2>&1 | tee "${local_log_dir}/run.log"
            then
                echo "  [PASS] ${task_suite} / ${DOMAIN_NAME} ✅" | tee -a "${SUMMARY}"
                PASSED=$((PASSED + 1))
            else
                echo "  [FAIL] ${task_suite} / ${DOMAIN_NAME} ❌" | tee -a "${SUMMARY}"
                FAILED=$((FAILED + 1))
            fi
        done
    done

    stop_server
done

# ═══════════════════════════  汇总  ═══════════════════════════════════════════

echo "" | tee -a "${SUMMARY}"
echo "══════════════════════════════════════════════════════════" | tee -a "${SUMMARY}"
echo "  完成时间: $(date)" | tee -a "${SUMMARY}"
echo "  总计: ${TOTAL}  通过: ${PASSED}  失败: ${FAILED}  跳过: ${SKIPPED}" | tee -a "${SUMMARY}"
echo "══════════════════════════════════════════════════════════" | tee -a "${SUMMARY}"

# ── 调用 Python 汇总所有 eval_results.json 生成表格 ──
echo "" | tee -a "${SUMMARY}"
echo "▶ 生成汇总表格 ..." | tee -a "${SUMMARY}"

/.venv_libero/bin/python /app/eval/aggregate_results.py \
    --log-root "${LOG_ROOT}" \
    --output "${RESULTS_ROOT}/results_table_${TIMESTAMP}.csv" \
    2>&1 | tee -a "${SUMMARY}"

echo ""
echo "批量摘要文件: ${SUMMARY}"
echo "结果 CSV:     ${RESULTS_ROOT}/results_table_${TIMESTAMP}.csv"