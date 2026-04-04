#!/usr/bin/env bash
# =============================================================================
# RunPod 首次初始化脚本
# 在 RunPod pod 内通过 SSH 运行（此时 /app 已挂载 Network Volume，但内容为空）
# 使用方法:
#   bash /tmp/runpod_init.sh https://github.com/yourname/Sim2Sim_Lab.git
# =============================================================================
set -euo pipefail

REPO_URL="${1:-}"
MOUNT_PATH="/app"

if [ -z "${REPO_URL}" ]; then
    echo "用法: bash runpod_init.sh <git仓库地址>"
    echo "示例: bash runpod_init.sh https://github.com/yourname/Sim2Sim_Lab.git"
    exit 1
fi

echo "[init] 检查挂载点 ${MOUNT_PATH} ..."
if [ ! -d "${MOUNT_PATH}" ]; then
    echo "错误: ${MOUNT_PATH} 不存在，请确认 Network Volume 已正确挂载。"
    exit 1
fi

# 如果目录为空，克隆项目
if [ -z "$(ls -A ${MOUNT_PATH})" ]; then
    echo "[init] 克隆项目到 ${MOUNT_PATH} ..."
    git clone --recurse-submodules "${REPO_URL}" "${MOUNT_PATH}"
else
    echo "[init] ${MOUNT_PATH} 非空，跳过克隆，仅更新子模块 ..."
    cd "${MOUNT_PATH}"
    git submodule update --init --recursive
fi

# 创建模型缓存目录
mkdir -p "${MOUNT_PATH}/.cache/openpi"
mkdir -p "${MOUNT_PATH}/data/libero/videos"

echo ""
echo "[init] 初始化完成！项目结构:"
ls -la "${MOUNT_PATH}/"
echo ""
echo "接下来可以运行评估："
echo "  bash /app/docker/entrypoint.sh"