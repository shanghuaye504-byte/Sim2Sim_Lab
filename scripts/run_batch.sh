##!/usr/bin/env bash
#set -euo pipefail
#
###############################################################################
##  Sim2Sim_Lab — Batch Experiment Script  run_batch.sh
##
##  Usage:
##    bash /app/eval/run_batch.sh
##
##  Override defaults via environment variables, e.g.:
##    NUM_TRIALS=10 bash /app/eval/run_batch.sh
###############################################################################
#
## ═══════════════════════════  Configuration  ═══════════════════════════════════
#
#
#
## Model checkpoints, format: "policy_config|checkpoint_dir"
## To add more checkpoints, simply add new lines
#CHECKPOINTS=(
#    "pi05_libero|gs://openpi-assets/checkpoints/pi05_libero"
#    # "another_policy|/app/data/base_checkpoints/another_policy"
#)
#
## Task suites
#TASK_SUITES=(
#    "libero_spatial"
#    "libero_object"
#    "libero_goal"
#    "libero_10"
#)
#
## Domain config directory
#DOMAIN_CONFIG_DIR="/app/eval/domain_configs"
#
## Number of trials per task
#NUM_TRIALS="${NUM_TRIALS:-20}"
#
## Server port
#SERVER_PORT="${SERVER_PORT:-8000}"
#
## Output root directory
#RESULTS_ROOT="${RESULTS_ROOT:-/app/data/libero}"
#VIDEO_ROOT="${RESULTS_ROOT}/videos"
#LOG_ROOT="${RESULTS_ROOT}/logs"
#
## Server start timeout (seconds)
#SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"
#
## MuJoCo rendering
#MUJOCO_GL="${MUJOCO_GL:-egl}"
#
## ═══════════════════════════  Functions  ═══════════════════════════════════════
#
#SERVER_PID=""
#
#start_server() {
#    local policy_config="$1"
#    local policy_dir="$2"
#
#    echo "[server] Starting policy server: config=${policy_config}, dir=${policy_dir}"
#
#    PYTHONPATH=/app/third_party/openpi/src:/app/third_party/openpi/packages/openpi-client/src:/app/src \
#    OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/app/.cache/openpi}" \
#        /.venv/bin/python \
#            /app/third_party/openpi/scripts/serve_policy.py \
#            --port "${SERVER_PORT}" \
#            policy:checkpoint \
#            --policy.config "${policy_config}" \
#            --policy.dir "${policy_dir}" &
#    SERVER_PID=$!
#
#    local elapsed=0
#    until curl -sf "http://127.0.0.1:${SERVER_PORT}/healthz" > /dev/null 2>&1; do
#        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
#            echo "[server] ❌ Policy server exited unexpectedly"; return 1
#        fi
#        if [ "${elapsed}" -ge "${SERVER_START_TIMEOUT}" ]; then
#            echo "[server] ❌ Start timeout (${SERVER_START_TIMEOUT}s)"
#            kill "${SERVER_PID}" 2>/dev/null; return 1
#        fi
#        sleep 5; elapsed=$((elapsed + 5))
#        [ $((elapsed % 30)) -eq 0 ] && echo "[server] Waited ${elapsed}s ..."
#    done
#    echo "[server] ✅ Server ready (PID=${SERVER_PID})"
#}
#
#stop_server() {
#    if [ -n "${SERVER_PID}" ]; then
#        echo "[server] Stopping policy server (PID=${SERVER_PID})"
#        kill "${SERVER_PID}" 2>/dev/null || true
#        wait "${SERVER_PID}" 2>/dev/null || true
#        SERVER_PID=""
#    fi
#}
#
#run_single_eval() {
#    local task_suite="$1"
#    local domain_config_file="$2"  # empty string means source domain
#    local domain_name="$3"
#    local model_name="$4"
#
#    local log_dir="${LOG_ROOT}/${model_name}/${task_suite}/${domain_name}"
#    local video_dir="${VIDEO_ROOT}/${model_name}/${task_suite}/${domain_name}"
#    mkdir -p "${log_dir}" "${video_dir}"
#
#    MUJOCO_GL="${MUJOCO_GL}" \
#    MUJOCO_EGL_DEVICE_ID=0 \
#    PYOPENGL_PLATFORM=egl \
#    LIBERO_CONFIG_PATH=/opt/libero_config \
#    PYTHONPATH=/app/third_party/openpi/third_party/libero \
#    DOMAIN_CONFIG_FILE="${domain_config_file}" \
#        /.venv_libero/bin/python \
#            /app/eval/domain_eval.py \
#            --args.host 127.0.0.1 \
#            --args.port "${SERVER_PORT}" \
#            --args.task-suite-name "${task_suite}" \
#            --args.num-trials-per-task "${NUM_TRIALS}" \
#            --args.video-out-path "${video_dir}" \
#            --args.log-dir "${log_dir}"
#}
#
## ═══════════════════════════  Build Domain List  ═══════════════════════════════
#
## Three intensity levels
#DOMAIN_LEVELS=("weak" "medium" "strong")
#
#declare -a DOMAIN_ENTRIES=("source||")   # source domain: no config file, no level
#
#if [ -d "${DOMAIN_CONFIG_DIR}" ]; then
#    for f in "${DOMAIN_CONFIG_DIR}"/*.yaml; do
#        [ -f "$f" ] || continue
#        base_name="$(basename "$f" .yaml)"
#
#        # Detect multi-level format (contains "levels:" keyword)
#        if grep -q "^levels:" "$f"; then
#            for lvl in "${DOMAIN_LEVELS[@]}"; do
#                DOMAIN_ENTRIES+=("${base_name}_${lvl}|${f}|${lvl}")
#            done
#        else
#            # Backward compatible with flat format
#            DOMAIN_ENTRIES+=("${base_name}|${f}|")
#        fi
#    done
#fi
#
## ═══════════════════════════  Compute Total Experiments  ═══════════════════════
#
#TOTAL_PLANNED=$(( ${#CHECKPOINTS[@]} * ${#TASK_SUITES[@]} * ${#DOMAIN_ENTRIES[@]} ))
#
## ═══════════════════════════  Main Loop  ═══════════════════════════════════════
#
#TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
#TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
#SUMMARY="${RESULTS_ROOT}/batch_summary_${TIMESTAMP}.log"
#mkdir -p "${RESULTS_ROOT}"
#
## Ensure server is stopped when script exits
#trap stop_server EXIT
#
#{
#echo "══════════════════════════════════════════════════════════"
#echo "  Sim2Sim_Lab — Batch Experiments"
#echo "  Start time        : $(date)"
#echo "  Checkpoints       : ${#CHECKPOINTS[@]}"
#echo "  Task suites       : ${TASK_SUITES[*]}"
#echo "  Domains           : ${#DOMAIN_ENTRIES[@]} (incl. source)"
#echo "  Trials per task   : ${NUM_TRIALS}"
#echo "  Planned total     : ${TOTAL_PLANNED}"
#echo "══════════════════════════════════════════════════════════"
#echo ""
#} | tee "${SUMMARY}"
#
#for ckpt_entry in "${CHECKPOINTS[@]}"; do
#    IFS='|' read -r POLICY_CONFIG POLICY_DIR <<< "${ckpt_entry}"
#    MODEL_NAME="$(basename "${POLICY_DIR}")"
#
#    echo "" | tee -a "${SUMMARY}"
#    echo "▶ Model: ${MODEL_NAME} (config=${POLICY_CONFIG})" | tee -a "${SUMMARY}"
#    echo "  Path: ${POLICY_DIR}" | tee -a "${SUMMARY}"
#    echo "" | tee -a "${SUMMARY}"
#
#    # Start server once per checkpoint
#    if ! start_server "${POLICY_CONFIG}" "${POLICY_DIR}"; then
#        echo "  [FATAL] Cannot start server, skipping this checkpoint" | tee -a "${SUMMARY}"
#        # Mark all experiments under this checkpoint as FAILED
#        local_fail_count=$(( ${#TASK_SUITES[@]} * ${#DOMAIN_ENTRIES[@]} ))
#        TOTAL=$((TOTAL + local_fail_count))
#        FAILED=$((FAILED + local_fail_count))
#        continue
#    fi
#
#    for task_suite in "${TASK_SUITES[@]}"; do
#        for domain_entry in "${DOMAIN_ENTRIES[@]}"; do
#            IFS='|' read -r DOMAIN_NAME DOMAIN_CONFIG_FILE <<< "${domain_entry}"
#            TOTAL=$((TOTAL + 1))
#
#            # ── Resume: skip if JSON already exists ──
#            json_path="${LOG_ROOT}/${MODEL_NAME}/${task_suite}/${DOMAIN_NAME}/eval_results.json"
#            if [ -f "${json_path}" ]; then
#                echo "  [SKIP] ${task_suite} / ${DOMAIN_NAME}" | tee -a "${SUMMARY}"
#                SKIPPED=$((SKIPPED + 1))
#                continue
#            fi
#
#            echo "  [RUN]  ${task_suite} / ${DOMAIN_NAME}  (${TOTAL}/${TOTAL_PLANNED})" | tee -a "${SUMMARY}"
#
#            # Ensure log directory exists (needed for tee)
#            local_log_dir="${LOG_ROOT}/${MODEL_NAME}/${task_suite}/${DOMAIN_NAME}"
#            mkdir -p "${local_log_dir}"
#
#            if run_single_eval \
#                "${task_suite}" \
#                "${DOMAIN_CONFIG_FILE}" \
#                "${DOMAIN_NAME}" \
#                "${MODEL_NAME}" \
#                2>&1 | tee "${local_log_dir}/run.log"
#            then
#                echo "  [PASS] ${task_suite} / ${DOMAIN_NAME} ✅" | tee -a "${SUMMARY}"
#                PASSED=$((PASSED + 1))
#            else
#                echo "  [FAIL] ${task_suite} / ${DOMAIN_NAME} ❌" | tee -a "${SUMMARY}"
#                FAILED=$((FAILED + 1))
#            fi
#        done
#    done
#
#    stop_server
#done
#
## ═══════════════════════════  Summary  ═════════════════════════════════════════
#
#echo "" | tee -a "${SUMMARY}"
#echo "══════════════════════════════════════════════════════════" | tee -a "${SUMMARY}"
#echo "  Finished: $(date)" | tee -a "${SUMMARY}"
#echo "  Total: ${TOTAL}  Passed: ${PASSED}  Failed: ${FAILED}  Skipped: ${SKIPPED}" | tee -a "${SUMMARY}"
#echo "══════════════════════════════════════════════════════════" | tee -a "${SUMMARY}"
#
## ── Call Python to aggregate all eval_results.json into a table ──
#echo "" | tee -a "${SUMMARY}"
#echo "▶ Generating summary table ..." | tee -a "${SUMMARY}"
#
#/.venv_libero/bin/python /app/eval/aggregate_results.py \
#    --log-root "${LOG_ROOT}" \
#    --output "${RESULTS_ROOT}/results_table_${TIMESTAMP}.csv" \
#    2>&1 | tee -a "${SUMMARY}"
#
#echo ""
#echo "Batch summary file: ${SUMMARY}"
#echo "Results CSV:        ${RESULTS_ROOT}/results_table_${TIMESTAMP}.csv"


#!/usr/bin/env bash
set -euo pipefail

##############################################################################
#  Sim2Sim_Lab — Batch Experiment Script  run_batch.sh
#
#  Usage:
#    bash /app/eval/run_batch.sh
#
#  Override defaults via environment variables, e.g.:
#    NUM_TRIALS=10 bash /app/eval/run_batch.sh
##############################################################################

# ═══════════════════════════  Configuration  ═══════════════════════════════════

# Model checkpoints, format: "policy_config|checkpoint_dir"
# To add more checkpoints, simply add new lines
CHECKPOINTS=(
    "pi05_libero|gs://openpi-assets/checkpoints/pi05_libero"
    # "another_policy|/app/data/base_checkpoints/another_policy"
)

# Task suites
TASK_SUITES=(
    "libero_spatial"
    "libero_object"
    "libero_goal"
    "libero_10"
)

# Domain config directory
DOMAIN_CONFIG_DIR="/app/eval/domain_configs"

# Number of trials per task
NUM_TRIALS="${NUM_TRIALS:-20}"

# Server port
SERVER_PORT="${SERVER_PORT:-8000}"

# Output root directory
RESULTS_ROOT="${RESULTS_ROOT:-/app/data/libero}"
VIDEO_ROOT="${RESULTS_ROOT}/videos"
LOG_ROOT="${RESULTS_ROOT}/logs"

# Server start timeout (seconds)
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-600}"

# MuJoCo rendering
MUJOCO_GL="${MUJOCO_GL:-egl}"

# ═══════════════════════════  Functions  ═══════════════════════════════════════

SERVER_PID=""

start_server() {
    local policy_config="$1"
    local policy_dir="$2"

    echo "[server] Starting policy server: config=${policy_config}, dir=${policy_dir}"

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
            echo "[server] ❌ Policy server exited unexpectedly"; return 1
        fi
        if [ "${elapsed}" -ge "${SERVER_START_TIMEOUT}" ]; then
            echo "[server] ❌ Start timeout (${SERVER_START_TIMEOUT}s)"
            kill "${SERVER_PID}" 2>/dev/null; return 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        [ $((elapsed % 30)) -eq 0 ] && echo "[server] Waited ${elapsed}s ..."
    done
    echo "[server] ✅ Server ready (PID=${SERVER_PID})"
}

stop_server() {
    if [ -n "${SERVER_PID}" ]; then
        echo "[server] Stopping policy server (PID=${SERVER_PID})"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}

# ★ Change 2: run_single_eval now takes a 5th argument domain_level, passed to DOMAIN_LEVEL env variable
run_single_eval() {
    local task_suite="$1"
    local domain_config_file="$2"  # empty string means source domain
    local domain_name="$3"
    local model_name="$4"
    local domain_level="$5"        # empty string means flat format or source

    local log_dir="${LOG_ROOT}/${model_name}/${task_suite}/${domain_name}"
    local video_dir="${VIDEO_ROOT}/${model_name}/${task_suite}/${domain_name}"
    mkdir -p "${log_dir}" "${video_dir}"

    MUJOCO_GL="${MUJOCO_GL}" \
    MUJOCO_EGL_DEVICE_ID=0 \
    PYOPENGL_PLATFORM=egl \
    LIBERO_CONFIG_PATH=/opt/libero_config \
    PYTHONPATH=/app/third_party/openpi/third_party/libero \
    DOMAIN_CONFIG_FILE="${domain_config_file}" \
    DOMAIN_LEVEL="${domain_level}" \
        /.venv_libero/bin/python \
            /app/eval/domain_eval.py \
            --args.host 127.0.0.1 \
            --args.port "${SERVER_PORT}" \
            --args.task-suite-name "${task_suite}" \
            --args.num-trials-per-task "${NUM_TRIALS}" \
            --args.video-out-path "${video_dir}" \
            --args.log-dir "${log_dir}"
}

# ═══════════════════════════  Build Domain List  ═══════════════════════════════

# Three intensity levels
DOMAIN_LEVELS=("weak" "medium" "strong")

# Format: "display_name|config_file_path|level"
# source domain: no config file, no level
declare -a DOMAIN_ENTRIES=("source||")

if [ -d "${DOMAIN_CONFIG_DIR}" ]; then
    for f in "${DOMAIN_CONFIG_DIR}"/*.yaml; do
        [ -f "$f" ] || continue
        base_name="$(basename "$f" .yaml)"

        # Detect multi-level format (contains "levels:" keyword)
        if grep -q "^levels:" "$f"; then
            for lvl in "${DOMAIN_LEVELS[@]}"; do
                DOMAIN_ENTRIES+=("${base_name}_${lvl}|${f}|${lvl}")
            done
        else
            # Backward compatible with flat format
            DOMAIN_ENTRIES+=("${base_name}|${f}|")
        fi
    done
fi

# ═══════════════════════════  Compute Total Experiments  ═══════════════════════

TOTAL_PLANNED=$(( ${#CHECKPOINTS[@]} * ${#TASK_SUITES[@]} * ${#DOMAIN_ENTRIES[@]} ))

# ═══════════════════════════  Main Loop  ═══════════════════════════════════════

TOTAL=0; PASSED=0; FAILED=0; SKIPPED=0
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_ROOT}/batch_summary_${TIMESTAMP}.log"
mkdir -p "${RESULTS_ROOT}"

# Ensure server is stopped when script exits
trap stop_server EXIT

{
echo "══════════════════════════════════════════════════════════"
echo "  Sim2Sim_Lab — Batch Experiments"
echo "  Start time        : $(date)"
echo "  Checkpoints       : ${#CHECKPOINTS[@]}"
echo "  Task suites       : ${TASK_SUITES[*]}"
echo "  Domains           : ${#DOMAIN_ENTRIES[@]} (incl. source)"
echo "  Trials per task   : ${NUM_TRIALS}"
echo "  Planned total     : ${TOTAL_PLANNED}"
echo "══════════════════════════════════════════════════════════"
echo ""
} | tee "${SUMMARY}"

for ckpt_entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r POLICY_CONFIG POLICY_DIR <<< "${ckpt_entry}"
    MODEL_NAME="$(basename "${POLICY_DIR}")"

    echo "" | tee -a "${SUMMARY}"
    echo "▶ Model: ${MODEL_NAME} (config=${POLICY_CONFIG})" | tee -a "${SUMMARY}"
    echo "  Path: ${POLICY_DIR}" | tee -a "${SUMMARY}"
    echo "" | tee -a "${SUMMARY}"

    # Start server once per checkpoint
    if ! start_server "${POLICY_CONFIG}" "${POLICY_DIR}"; then
        echo "  [FATAL] Cannot start server, skipping this checkpoint" | tee -a "${SUMMARY}"
        # Mark all experiments under this checkpoint as FAILED
        local_fail_count=$(( ${#TASK_SUITES[@]} * ${#DOMAIN_ENTRIES[@]} ))
        TOTAL=$((TOTAL + local_fail_count))
        FAILED=$((FAILED + local_fail_count))
        continue
    fi

    for task_suite in "${TASK_SUITES[@]}"; do
        for domain_entry in "${DOMAIN_ENTRIES[@]}"; do
            # ★ Change 3: parse 3 fields (previously only parsed 2)
            IFS='|' read -r DOMAIN_NAME DOMAIN_CONFIG_FILE DOMAIN_LEVEL <<< "${domain_entry}"
            TOTAL=$((TOTAL + 1))

            # ── Resume: skip if JSON already exists ──
            json_path="${LOG_ROOT}/${MODEL_NAME}/${task_suite}/${DOMAIN_NAME}/eval_results.json"
            if [ -f "${json_path}" ]; then
                echo "  [SKIP] ${task_suite} / ${DOMAIN_NAME}" | tee -a "${SUMMARY}"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            echo "  [RUN]  ${task_suite} / ${DOMAIN_NAME}  (${TOTAL}/${TOTAL_PLANNED})" | tee -a "${SUMMARY}"

            # Ensure log directory exists (needed for tee)
            local_log_dir="${LOG_ROOT}/${MODEL_NAME}/${task_suite}/${DOMAIN_NAME}"
            mkdir -p "${local_log_dir}"

            if run_single_eval \
                "${task_suite}" \
                "${DOMAIN_CONFIG_FILE}" \
                "${DOMAIN_NAME}" \
                "${MODEL_NAME}" \
                "${DOMAIN_LEVEL}" \
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

# ═══════════════════════════  Summary  ═════════════════════════════════════════

echo "" | tee -a "${SUMMARY}"
echo "══════════════════════════════════════════════════════════" | tee -a "${SUMMARY}"
echo "  Finished: $(date)" | tee -a "${SUMMARY}"
echo "  Total: ${TOTAL}  Passed: ${PASSED}  Failed: ${FAILED}  Skipped: ${SKIPPED}" | tee -a "${SUMMARY}"
echo "══════════════════════════════════════════════════════════" | tee -a "${SUMMARY}"

# ── Call Python to aggregate all eval_results.json into a table ──
echo "" | tee -a "${SUMMARY}"
echo "▶ Generating summary table ..." | tee -a "${SUMMARY}"

/.venv_libero/bin/python /app/eval/aggregate_results.py \
    --log-root "${LOG_ROOT}" \
    --output "${RESULTS_ROOT}/results_table_${TIMESTAMP}.csv" \
    2>&1 | tee -a "${SUMMARY}"

echo ""
echo "Batch summary file: ${SUMMARY}"
echo "Results CSV:        ${RESULTS_ROOT}/results_table_${TIMESTAMP}.csv"
