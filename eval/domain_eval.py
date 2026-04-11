#!/usr/bin/env python3
"""
Domain Shift Wrapper for LIBERO Evaluation
===========================================

核心设计:
  1. monkey-patch main._get_libero_env，注入 DomainShiftEnvWrapper
  2. Wrapper 在每次 env.reset() 后自动重新应用 domain shift
     （因为 robosuite hard_reset=True 每次 reset 都重建 sim，模型参数全部丢失）
  3. set_init_state() 不重建 sim，只设置状态并重新渲染，
     domain shift 在 reset 后持续生效直到下一次 reset

支持的 Domain Shift 类型:
  A. Lighting  — intensity, direction, warm/cool color, shadow, active
  B. Camera    — position offset, euler rotation, fovy
  C. Friction  — global, per-geom keyword
  D. Material  — specular, shininess, reflectance (global + per-material keyword)
  E. Geom RGBA — per-geom keyword (保留原有功能)

使用方式:
    DOMAIN_CONFIG_FILE=/path/to/config.yaml \\
    python /app/eval/domain_eval.py \\
        --args.task-suite-name libero_spatial \\
        --args.num-trials-per-task 50
"""

import dataclasses
import logging
import os
import sys

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("domain_eval")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: 把 main.py 所在目录加入 sys.path，导入 libero_main
# ═══════════════════════════════════════════════════════════════════════════════
LIBERO_MAIN_DIR = os.environ.get(
    "LIBERO_MAIN_DIR",
    "/app/third_party/openpi/examples/libero",
)
sys.path.insert(0, LIBERO_MAIN_DIR)

import main as libero_main  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: 扩展 Args，新增 log_dir 字段（bash 脚本会传 --args.log-dir）
#         原始 main.py 的 Args 没有此字段，不扩展的话 tyro 会报错
# ═══════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class DomainEvalArgs(libero_main.Args):
    log_dir: str = "data/libero/logs"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: 读取 YAML domain config
# ═══════════════════════════════════════════════════════════════════════════════
def _load_domain_config() -> dict:
    config_file = os.environ.get("DOMAIN_CONFIG_FILE", "").strip()
    if not config_file:
        log.info("[domain_eval] DOMAIN_CONFIG_FILE 未设置，以 source domain 运行。")
        return {}
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f) or {}
    log.info(f"[domain_eval] 已加载 domain config: {config_file}")
    log.info(f"[domain_eval] 配置内容:\n{yaml.dump(cfg, default_flow_style=False)}")
    return cfg


DOMAIN_CONFIG = _load_domain_config()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: MuJoCo 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def _find_sim(env):
    """
    在 LIBERO 多层包装中查找 robosuite 的 sim 对象。
    OffScreenRenderEnv → ControlEnv.env → robosuite env.sim
    ControlEnv 自身也有 @property sim 代理到 self.env.sim
    """
    for path in ("sim", "env.sim", "env.env.sim"):
        obj = env
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None:
                return obj
        except AttributeError:
            continue
    log.warning("[domain_eval] 未能找到 sim 对象！请检查 LIBERO/robosuite 版本。")
    return None


def _get_raw_model(model):
    """
    robosuite 的 MjSim.model 是对 mujoco.MjModel 的包装，
    mujoco C API（如 mj_name2id）需要原始 model 对象。
    """
    return getattr(model, "_model", model)


def _id2name(model, type_enum, idx):
    """根据 ID 获取 MuJoCo 对象名称（兼容 robosuite 包装）"""
    import mujoco
    try:
        return mujoco.mj_id2name(_get_raw_model(model), type_enum, idx) or ""
    except Exception:
        return ""


def _name2id(model, type_enum, name):
    """根据名称获取 MuJoCo 对象 ID（兼容 robosuite 包装）"""
    import mujoco
    try:
        # 优先用 robosuite 包装的方法（camera 用 camera_name2id 等）
        type_str_map = {
            mujoco.mjtObj.mjOBJ_CAMERA: "camera",
            mujoco.mjtObj.mjOBJ_GEOM: "geom",
            mujoco.mjtObj.mjOBJ_LIGHT: "light",
            mujoco.mjtObj.mjOBJ_MATERIAL: "mat",
        }
        wrapper_method = f"{type_str_map.get(type_enum, '')}_name2id"
        fn = getattr(model, wrapper_method, None)
        if fn is not None:
            return fn(name)
    except Exception:
        pass
    # fallback: 直接调 mujoco C API
    try:
        result = mujoco.mj_name2id(_get_raw_model(model), type_enum, name)
        return result if result >= 0 else -1
    except Exception:
        return -1


def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    """
    欧拉角（度）→ MuJoCo 四元数 [w, x, y, z]
    采用 ZYX 内旋约定：依次绕 X(roll) → Y(pitch) → Z(yaw)
    """
    r = np.radians(roll_deg) / 2.0
    p = np.radians(pitch_deg) / 2.0
    y = np.radians(yaw_deg) / 2.0

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = np.array([w, x, yy, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_mul(q1, q2):
    """四元数乘法 q1 × q2，格式均为 [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Domain Shift 核心应用函数
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_domain_shift(env, config: dict, verbose: bool = False) -> None:
    """
    在 env.reset() 完成后修改 MuJoCo model 属性。

    参数:
        env:     OffScreenRenderEnv 实例（ControlEnv 子类）
        config:  从 YAML 加载的 domain shift 配置字典
        verbose: 首次调用时为 True，打印详细的匹配信息

    为什么需要在每次 reset 后重新调用？
        robosuite 默认 hard_reset=True，每次 env.reset() 都会执行
        _load_model() + _initialize_sim()，完全重建 MjSim 对象，
        之前对 model 的所有修改全部丢失。
    """
    import mujoco

    if not config:
        return

    sim = _find_sim(env)
    if sim is None:
        return

    model = sim.model

    # ── 如果 verbose，打印场景概览 ────────────────────────────────────────────
    if verbose:
        log.info(
            f"[domain_eval] 场景概览: "
            f"ngeom={model.ngeom}, nlight={model.nlight}, "
            f"ncam={model.ncam}, nmat={model.nmat}"
        )
        # 打印所有相机名称，方便用户确认
        for i in range(model.ncam):
            cname = _id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            log.info(f"  camera[{i}] = '{cname}', "
                     f"pos={model.cam_pos[i]}, fovy={model.cam_fovy[i]:.1f}")
        # 打印所有光源名称
        for i in range(model.nlight):
            lname = _id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i)
            log.info(f"  light[{i}] = '{lname}', "
                     f"diffuse={model.light_diffuse[i]}, "
                     f"dir={model.light_dir[i]}")

    # ══════════════════════════════════════════════════════════════════════════
    # A. 光照调整 (Lighting)
    # ══════════════════════════════════════════════════════════════════════════
    lighting = config.get("lighting", {})
    if lighting and model.nlight > 0:
        if verbose:
            log.info(f"[domain_eval][A] 应用光照调整（{model.nlight} 个光源）: {lighting}")

        # A1. 漫反射强度缩放（控制整体明暗）
        if "diffuse_scale" in lighting:
            s = float(lighting["diffuse_scale"])
            model.light_diffuse[:] = np.clip(model.light_diffuse * s, 0.0, 1.0)

        # A2. 镜面反射强度缩放（控制高光强弱）
        if "specular_scale" in lighting:
            s = float(lighting["specular_scale"])
            model.light_specular[:] = np.clip(model.light_specular * s, 0.0, 1.0)

        # A3. 环境光强度缩放（控制阴影区域的亮度下限）
        if "ambient_scale" in lighting:
            s = float(lighting["ambient_scale"])
            model.light_ambient[:] = np.clip(model.light_ambient * s, 0.0, 1.0)

        # A4. 光照方向偏移（改变光线入射角度）
        #     偏移后重新归一化为单位向量
        if "direction_offset" in lighting:
            offset = np.array(lighting["direction_offset"], dtype=np.float64)
            for i in range(model.nlight):
                d = model.light_dir[i] + offset
                norm = np.linalg.norm(d)
                if norm > 1e-8:
                    model.light_dir[i] = d / norm

        # A5. 光源位置偏移
        if "position_offset" in lighting:
            offset = np.array(lighting["position_offset"], dtype=np.float64)
            model.light_pos[:] = model.light_pos + offset

        # A6. 色温偏移：正 R / 负 B = 暖色；负 R / 正 B = 冷色
        #     加到 diffuse 上，影响直射光颜色
        if "color_shift" in lighting:
            shift = np.array(lighting["color_shift"], dtype=np.float64)
            model.light_diffuse[:] = np.clip(model.light_diffuse + shift, 0.0, 1.0)

        # A7. 阴影开关（0 = 不投射阴影，1 = 投射阴影）
        if "castshadow" in lighting:
            model.light_castshadow[:] = int(lighting["castshadow"])

        # A8. 光源激活开关（0 = 关闭，1 = 开启）
        #     注意：关闭所有光源会导致场景全黑
        if "active" in lighting:
            model.light_active[:] = int(lighting["active"])

    # ══════════════════════════════════════════════════════════════════════════
    # B. 相机调整 (Camera)
    # ══════════════════════════════════════════════════════════════════════════
    camera = config.get("camera", {})
    cam_shifts = camera.get("shifts", [])
    if cam_shifts and model.ncam > 0:
        if verbose:
            log.info(f"[domain_eval][B] 应用相机调整（{len(cam_shifts)} 项配置）")

        for shift in cam_shifts:
            cam_name = shift.get("name", "")
            cam_id = _name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                log.warning(f"[domain_eval][B] 未找到相机 '{cam_name}'，跳过。"
                            f"可用相机: {[_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]}")
                continue

            # B1. 位置偏移 [dx, dy, dz]（米）
            #     agentview 等固定相机：世界坐标偏移
            #     eye_in_hand 等附着相机：相对父体偏移
            if "pos_offset" in shift:
                offset = np.array(shift["pos_offset"], dtype=np.float64)
                model.cam_pos[cam_id] += offset
                if verbose:
                    log.info(f"  cam '{cam_name}': pos += {offset} → {model.cam_pos[cam_id]}")

            # B2. 欧拉角偏移 [roll, pitch, yaw]（度）
            #     将偏移四元数左乘到原始四元数上
            if "euler_offset_deg" in shift:
                euler = shift["euler_offset_deg"]
                dq = _euler_deg_to_quat(euler[0], euler[1], euler[2])
                orig_q = model.cam_quat[cam_id].copy()
                new_q = _quat_mul(dq, orig_q)
                new_q /= np.linalg.norm(new_q)  # 保证单位四元数
                model.cam_quat[cam_id] = new_q
                if verbose:
                    log.info(f"  cam '{cam_name}': euler offset {euler}° applied")

            # B3. 视场角偏移（度），裁剪到 [10, 160]
            if "fovy_offset" in shift:
                fov_offset = float(shift["fovy_offset"])
                old_fov = float(model.cam_fovy[cam_id])
                model.cam_fovy[cam_id] = np.clip(old_fov + fov_offset, 10.0, 160.0)
                if verbose:
                    log.info(f"  cam '{cam_name}': fovy {old_fov:.1f} → {float(model.cam_fovy[cam_id]):.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # C. 摩擦力调整 (Friction)
    #    geom_friction 是 (ngeom, 3) 数组: [滑动摩擦, 扭转摩擦, 滚动摩擦]
    # ══════════════════════════════════════════════════════════════════════════
    friction = config.get("friction", {})
    if friction and model.ngeom > 0:
        if verbose:
            log.info(f"[domain_eval][C] 应用摩擦力调整（{model.ngeom} 个 geom）")

        # C1. 全局摩擦缩放（标量 → 三个分量统一缩放；列表 → 分别缩放）
        if "global_scale" in friction:
            gs = friction["global_scale"]
            if isinstance(gs, (int, float)):
                gs = [float(gs)] * 3
            gs = np.array(gs, dtype=np.float64)
            model.geom_friction[:] = np.maximum(model.geom_friction * gs, 0.0)
            if verbose:
                log.info(f"  全局摩擦缩放: ×{gs}")

        # C2. 按名称关键词缩放指定 geom 的摩擦
        geom_friction_shifts = friction.get("geom_friction_shifts", [])
        for gfs in geom_friction_shifts:
            keyword = gfs.get("name_contains", "").lower()
            if not keyword:
                continue
            fs = gfs.get("friction_scale", 1.0)
            if isinstance(fs, (int, float)):
                fs = [float(fs)] * 3
            fs = np.array(fs, dtype=np.float64)

            matched = 0
            for i in range(model.ngeom):
                gname = _id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i).lower()
                if keyword in gname:
                    model.geom_friction[i] = np.maximum(
                        model.geom_friction[i] * fs, 0.0
                    )
                    matched += 1
            if verbose:
                log.info(f"  摩擦关键词 '{keyword}': 匹配 {matched} 个 geom, scale={fs}")

    # ══════════════════════════════════════════════════════════════════════════
    # D. 材质光学属性调整 (Material Optical)
    # ══════════════════════════════════════════════════════════════════════════
    material_cfg = config.get("material", {})
    if material_cfg and model.nmat > 0:
        if verbose:
            log.info(f"[domain_eval][D] 应用材质光学调整（{model.nmat} 个材质）")

        # D1. 全局 specular 缩放
        if "global_specular_scale" in material_cfg:
            s = float(material_cfg["global_specular_scale"])
            model.mat_specular[:] = np.clip(model.mat_specular * s, 0.0, 1.0)
            if verbose:
                log.info(f"  全局 specular ×{s}")

        # D2. 全局 shininess 缩放
        if "global_shininess_scale" in material_cfg:
            s = float(material_cfg["global_shininess_scale"])
            model.mat_shininess[:] = np.clip(model.mat_shininess * s, 0.0, 1.0)
            if verbose:
                log.info(f"  全局 shininess ×{s}")

        # D3. 全局 reflectance 缩放
        if "global_reflectance_scale" in material_cfg:
            s = float(material_cfg["global_reflectance_scale"])
            model.mat_reflectance[:] = np.clip(model.mat_reflectance * s, 0.0, 1.0)
            if verbose:
                log.info(f"  全局 reflectance ×{s}")

        # D4. 按名称关键词设置指定材质的属性（绝对值，非缩放）
        material_shifts = material_cfg.get("material_shifts", [])
        for ms in material_shifts:
            keyword = ms.get("name_contains", "").lower()
            if not keyword:
                continue
            matched = 0
            for i in range(model.nmat):
                mname = _id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, i).lower()
                if keyword in mname:
                    if "specular" in ms:
                        model.mat_specular[i] = float(ms["specular"])
                    if "shininess" in ms:
                        model.mat_shininess[i] = float(ms["shininess"])
                    if "reflectance" in ms:
                        model.mat_reflectance[i] = float(ms["reflectance"])
                    if "rgba" in ms:
                        model.mat_rgba[i] = np.array(ms["rgba"], dtype=np.float32)
                    matched += 1
            if verbose:
                log.info(f"  材质关键词 '{keyword}': 匹配 {matched} 个材质")

    # ══════════════════════════════════════════════════════════════════════════
    # E. Geom RGBA 调整（保留原有功能，统一到新框架下）
    # ══════════════════════════════════════════════════════════════════════════
    geom_shifts = config.get("geom_rgba_shifts", [])
    if geom_shifts and model.ngeom > 0:
        if verbose:
            log.info(f"[domain_eval][E] 应用 geom RGBA 调整")
        for gs in geom_shifts:
            keyword = gs.get("name_contains", "").lower()
            rgba = gs.get("rgba", None)
            rgba_scale = gs.get("rgba_scale", None)
            if not keyword:
                continue
            matched = 0
            for i in range(model.ngeom):
                gname = _id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i).lower()
                if keyword in gname:
                    if rgba is not None:
                        model.geom_rgba[i] = np.array(rgba, dtype=np.float32)
                    if rgba_scale is not None:
                        model.geom_rgba[i, :3] = np.clip(
                            model.geom_rgba[i, :3] * float(rgba_scale), 0.0, 1.0
                        )
                    matched += 1
            if verbose:
                log.info(f"  geom_rgba 关键词 '{keyword}': 匹配 {matched} 个 geom")

    # ── 调用 forward() 使物理参数改动（摩擦等）传播到派生量 ────────────────────
    # 视觉参数（光照/相机/材质）在下次渲染时自动生效，
    # 但 forward() 会更新接触相关的派生量，对摩擦力改动有帮助
    sim.forward()

    if verbose:
        log.info("[domain_eval] ✓ Domain shift 应用完成。")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: 环境包装器 — 解决 hard_reset 导致修改丢失的核心问题
# ═══════════════════════════════════════════════════════════════════════════════

class DomainShiftEnvWrapper:
    """
    包装 OffScreenRenderEnv，在每次 reset() 后自动重新应用 domain shift。

    为什么需要这个包装器？
    ─────────────────────
    robosuite 默认 hard_reset=True，每次 env.reset() 都会：
      1. _load_model()      → 从 XML 重新构建模型
      2. _initialize_sim()  → 创建全新的 MjSim 对象
    这意味着之前对 model 参数（光照、相机、摩擦等）的任何修改全部丢失。

    评估循环中每个 episode 都调用 reset()，所以必须在每次 reset 后重新应用。

    调用顺序保证:
    ─────────────
    eval_libero 中的流程:
      env.reset()                              ← 重建 sim，我们在此后注入 shift
      obs = env.set_init_state(init_state)     ← 设置 qpos/qvel，调用 forward() + render
                                                  此时 domain shift 已经生效 ✓
      for step:
        obs, r, d, i = env.step(action)        ← 物理步进 + 渲染，shift 持续生效 ✓
    """

    def __init__(self, env, config: dict):
        # 使用 object.__setattr__ 避免触发 __getattr__
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_reset_count", 0)

    def reset(self):
        """调用原始 reset()，然后重新应用 domain shift。"""
        obs = self._env.reset()

        # 首次 reset 时 verbose=True，打印场景详细信息帮助调试
        verbose = (self._reset_count == 0)
        _apply_domain_shift(self._env, self._config, verbose=verbose)
        object.__setattr__(self, "_reset_count", self._reset_count + 1)

        return obs

    def set_init_state(self, init_state):
        """
        set_init_state 不重建 sim（只设置 qpos/qvel + forward + render），
        domain shift 在上一次 reset 中已经应用，此处自动生效。
        返回的 obs 已包含 domain shift 效果。
        """
        return self._env.set_init_state(init_state)

    def step(self, action):
        """直接转发，domain shift 在本 episode 内持续生效。"""
        return self._env.step(action)

    def seed(self, s):
        return self._env.seed(s)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        """未显式定义的属性/方法全部转发到底层 env。"""
        return getattr(self._env, name)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7: Monkey-patch _get_libero_env
# ═══════════════════════════════════════════════════════════════════════════════

_original_get_libero_env = libero_main._get_libero_env


def _patched_get_libero_env(task, resolution, seed):
    """
    调用原始函数创建 env，然后用 DomainShiftEnvWrapper 包装。

    注意：此时 env 还没有执行 reset()，domain shift 会在
    eval_libero() 中第一次调用 env.reset() 时自动应用。
    """
    env, task_description = _original_get_libero_env(task, resolution, seed)
    if DOMAIN_CONFIG:
        env = DomainShiftEnvWrapper(env, DOMAIN_CONFIG)
        log.info(f"[domain_eval] 环境已包装 DomainShiftEnvWrapper "
                 f"(task: {task_description})")
    return env, task_description


libero_main._get_libero_env = _patched_get_libero_env
log.info("[domain_eval] ✓ 已成功 patch _get_libero_env")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8: 入口
# ═══════════════════════════════════════════════════════════════════════════════

def _domain_eval_main(args: DomainEvalArgs) -> None:
    """
    薄包装层：扩展 Args 以支持 log_dir 字段，然后委托给 eval_libero。
    DomainEvalArgs 继承自 Args，所有原始字段完全兼容。
    """
    log.info(f"[domain_eval] log_dir = {args.log_dir}")
    log.info(f"[domain_eval] video_out_path = {args.video_out_path}")
    libero_main.eval_libero(args)


if __name__ == "__main__":
    import tyro
    tyro.cli(_domain_eval_main) 