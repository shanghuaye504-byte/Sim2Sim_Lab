#!/usr/bin/env python3
"""
Domain shift wrapper for LIBERO evaluation.
通过 monkey-patch main._get_libero_env 注入 domain shift，
完全不修改任何已有代码。

使用方式:
    DOMAIN_CONFIG_FILE=/path/to/dim_lighting.yaml \
    /.venv_libero/bin/python /app/eval/domain_eval.py \
        --args.task-suite-name libero_spatial \
        --args.num-trials-per-task 50
"""
import logging
import os
import sys

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("domain_eval")

# ── Step 1: 把 main.py 所在目录加入路径，然后 import 它 ──────────────────────
LIBERO_MAIN_DIR = os.environ.get(
    "LIBERO_MAIN_DIR",
    "/app/third_party/openpi/examples/libero"
)
sys.path.insert(0, LIBERO_MAIN_DIR)

import main as libero_main   # 此时 main.py 的模块级代码已执行完毕


# ── Step 2: 读取 domain config YAML ─────────────────────────────────────────
def _load_domain_config() -> dict:
    config_file = os.environ.get("DOMAIN_CONFIG_FILE", "").strip()
    if not config_file:
        log.info("[domain_eval] DOMAIN_CONFIG_FILE 未设置，以 source domain 运行。")
        return {}
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f) or {}
    log.info(f"[domain_eval] 已加载 domain config: {config_file}")
    log.info(f"[domain_eval] 配置内容: {cfg}")
    return cfg


DOMAIN_CONFIG = _load_domain_config()


# ── Step 3: 找到 MuJoCo sim 对象 ─────────────────────────────────────────────
def _find_sim(env):
    """
    LIBERO 的 OffScreenRenderEnv 是对 robosuite env 的包装，
    sim 对象可能在不同层级。依次尝试常见路径。
    """
    for path in ["sim", "env.sim", "env.env.sim", "_env.sim"]:
        obj = env
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None:
                log.info(f"[domain_eval] 找到 sim 对象，路径: env.{path}")
                return obj
        except AttributeError:
            continue
    log.warning("[domain_eval] 未能找到 sim 对象！请检查 LIBERO/robosuite 版本。")
    return None


# ── Step 4: 实际修改 MuJoCo 模型参数 ────────────────────────────────────────
def _apply_domain_shift(env, config: dict) -> None:
    """
    在 env 创建完成后修改 MuJoCo model 属性。
    这些属性是可写的 numpy 数组，修改后立即生效，
    且在同一个 env 对象的生命周期内（所有 episode）持续有效。
    """
    if not config:
        return

    sim = _find_sim(env)
    if sim is None:
        return

    model = sim.model

    # 打印所有 geom 名称，帮助你确认关键词
    log.info(f"ngeom={model.ngeom}, nlight={model.nlight}, nmat={model.nmat}")
    for i in range(model.ngeom):
        try:
            name = model.geom_id2name(i)
        except:
            name = f"geom_{i}"
        log.info(f"  geom[{i}] = {name}, rgba={model.geom_rgba[i]}")

    # ── 光照调整 ──────────────────────────────────────────────────────────
    lighting = config.get("lighting", {})
    if lighting and model.nlight > 0:
        log.info(f"[domain_eval] 应用光照调整: {lighting}（共 {model.nlight} 个光源）")

        # 漫反射强度缩放（影响整体明暗）
        if "diffuse_scale" in lighting:
            scale = float(lighting["diffuse_scale"])
            model.light_diffuse[:] = np.clip(model.light_diffuse * scale, 0.0, 1.0)

        # 镜面反射强度缩放（影响高光）
        if "specular_scale" in lighting:
            scale = float(lighting["specular_scale"])
            model.light_specular[:] = np.clip(model.light_specular * scale, 0.0, 1.0)

        # 环境光强度缩放
        if "ambient_scale" in lighting:
            scale = float(lighting["ambient_scale"])
            model.light_ambient[:] = np.clip(model.light_ambient * scale, 0.0, 1.0)

        # 是否投射阴影（0=不投，1=投）
        if "castshadow" in lighting:
            model.light_castshadow[:] = int(lighting["castshadow"])

    # ── 几何体颜色调整（geom_rgba）────────────────────────────────────────
    # 根据 geom 名称关键词批量修改 RGBA
    geom_shifts = config.get("geom_rgba_shifts", [])
    for shift in geom_shifts:
        keyword = shift.get("name_contains", "").lower()
        rgba = shift.get("rgba", None)
        rgba_scale = shift.get("rgba_scale", None)

        if not keyword:
            continue

        matched = 0
        for i in range(model.ngeom):
            # 兼容 mujoco-py 和新版 mujoco
            try:
                geom_name = (model.geom_id2name(i) or "").lower()
            except AttributeError:
                try:
                    import mujoco
                    geom_name = (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                    ).lower()
                except Exception:
                    geom_name = ""

            if keyword in geom_name:
                if rgba is not None:
                    model.geom_rgba[i] = np.array(rgba, dtype=np.float32)
                if rgba_scale is not None:
                    model.geom_rgba[i, :3] = np.clip(
                        model.geom_rgba[i, :3] * float(rgba_scale), 0.0, 1.0
                    )
                matched += 1

        log.info(f"[domain_eval] geom_rgba: 关键词 '{keyword}' 匹配到 {matched} 个 geom")

    # ── 材质属性调整（mat_rgba / 光泽度等）───────────────────────────────
    material_shifts = config.get("material_shifts", [])
    for shift in material_shifts:
        keyword = shift.get("name_contains", "").lower()
        rgba = shift.get("rgba", None)
        shininess = shift.get("shininess", None)
        specular = shift.get("specular", None)
        reflectance = shift.get("reflectance", None)

        if not keyword:
            continue

        matched = 0
        for i in range(model.nmat):
            try:
                mat_name = (model.mat_id2name(i) or "").lower()
            except AttributeError:
                try:
                    import mujoco
                    mat_name = (
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MAT, i) or ""
                    ).lower()
                except Exception:
                    mat_name = ""

            if keyword in mat_name:
                if rgba is not None:
                    model.mat_rgba[i] = np.array(rgba, dtype=np.float32)
                if shininess is not None:
                    model.mat_shininess[i] = float(shininess)
                if specular is not None:
                    model.mat_specular[i] = float(specular)
                if reflectance is not None:
                    model.mat_reflectance[i] = float(reflectance)
                matched += 1

        log.info(f"[domain_eval] material_shifts: 关键词 '{keyword}' 匹配到 {matched} 个 material")

    log.info("[domain_eval] Domain shift 应用完成。")


# ── Step 5: Monkey-patch _get_libero_env ────────────────────────────────────
#
# eval_libero() 在执行时会做 LOAD_GLOBAL '_get_libero_env'，
# 查找的是 main 模块自己的 __dict__，所以下面这行替换后，
# eval_libero 就会调用我们的包装版本，无需修改 main.py 任何一行。
#
_original_get_libero_env = libero_main._get_libero_env


def _patched_get_libero_env(task, resolution, seed):
    env, task_description = _original_get_libero_env(task, resolution, seed)
    _apply_domain_shift(env, DOMAIN_CONFIG)
    return env, task_description


libero_main._get_libero_env = _patched_get_libero_env
log.info("[domain_eval] 已成功 patch _get_libero_env。")


# ── Step 6: 入口——复用 main.py 的 Args 和 eval_libero，参数完全兼容 ──────────
if __name__ == "__main__":
    import tyro
    tyro.cli(libero_main.eval_libero)