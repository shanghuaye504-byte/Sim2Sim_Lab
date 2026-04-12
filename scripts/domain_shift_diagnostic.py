#!/usr/bin/env python3
"""
domain_shift_diagnostic.py
==========================
独立诊断脚本：验证 domain shift 是否真正影响了渲染输出。

输出:
    diagnostic_output/
    ├── diagnostic_original.png
    ├── diagnostic_shifted.png
    ├── diagnostic_diff.png       (差异 ×5 放大)
    └── 终端输出：像素差异统计 + 场景信息

用法:
    python domain_shift_diagnostic.py
"""

import os
import sys
import pathlib

import numpy as np

# ── 配置 ─────────────────────────────────────────────────────────────────────
LIBERO_MAIN_DIR = os.environ.get(
    "LIBERO_MAIN_DIR",
    "/app/third_party/openpi/examples/libero",
)
TASK_SUITE = "libero_spatial"
TASK_ID = 0
RESOLUTION = 256
SEED = 7
OUTPUT_DIR = "diagnostic_output"
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, LIBERO_MAIN_DIR)

import mujoco
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def create_env():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE]()
    task = task_suite.get_task(TASK_ID)
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
    )
    env.seed(SEED)
    init_states = task_suite.get_task_init_states(TASK_ID)
    return env, task.language, init_states


def find_sim(env):
    """尝试多条路径查找 sim"""
    for path in ("sim", "env.sim", "env.env.sim"):
        obj = env
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None:
                print(f"  ✓ sim found via: '{path}' → type={type(obj).__name__}")
                return obj
        except AttributeError:
            continue
    print("  ✗ sim NOT FOUND!")
    return None


def get_raw_model(model):
    """获取底层 mujoco.MjModel（绕过 robosuite wrapper）"""
    return getattr(model, "_model", model)


def id2name(raw_model, type_enum, idx):
    try:
        return mujoco.mj_id2name(raw_model, type_enum, idx) or f"<unnamed_{idx}>"
    except Exception:
        return f"<error_{idx}>"


def print_scene_info(sim):
    model = sim.model
    rm = get_raw_model(model)

    print(f"\n{'='*70}")
    print("  场景信息")
    print(f"{'='*70}")
    print(f"  model type     = {type(model).__name__}")
    print(f"  raw_model type = {type(rm).__name__}")
    print(f"  ngeom={rm.ngeom}, nlight={rm.nlight}, ncam={rm.ncam}, nmat={rm.nmat}")

    # ── Headlight（最关键）──
    print(f"\n  ◆ Headlight:")
    print(f"    active   = {rm.vis.headlight.active}")
    print(f"    diffuse  = {list(rm.vis.headlight.diffuse)}")
    print(f"    ambient  = {list(rm.vis.headlight.ambient)}")
    print(f"    specular = {list(rm.vis.headlight.specular)}")

    # ── 场景光源 ──
    print(f"\n  ◆ 场景光源 ({rm.nlight} 个):")
    if rm.nlight == 0:
        print("    ⚠️  没有场景光源！照明完全依赖 headlight。")
        print("    ⚠️  修改 model.light_* 不会有任何效果。")
    for i in range(rm.nlight):
        name = id2name(rm, mujoco.mjtObj.mjOBJ_LIGHT, i)
        print(f"    light[{i}] = '{name}'")
        print(f"      diffuse  = {list(model.light_diffuse[i])}")
        print(f"      dir      = {list(model.light_dir[i])}")
        print(f"      active   = {model.light_active[i] if hasattr(model, 'light_active') else 'N/A'}")

    # ── 相机 ──
    print(f"\n  ◆ 相机 ({rm.ncam} 个):")
    for i in range(rm.ncam):
        name = id2name(rm, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"    cam[{i}] = '{name}'")
        print(f"      pos  = {list(model.cam_pos[i])}")
        print(f"      fovy = {float(model.cam_fovy[i]):.1f}")


def get_image(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def apply_extreme_shift(sim):
    """应用极端 domain shift — 如果这都看不出区别，就是代码层面的问题"""
    model = sim.model
    rm = get_raw_model(model)

    print("\n应用极端 domain shift:")

    # ── 1. 关闭 headlight ──
    print("  ► 关闭 headlight (active=0, diffuse/ambient/specular=[0,0,0])")
    rm.vis.headlight.active = 0
    rm.vis.headlight.diffuse[:] = [0.0, 0.0, 0.0]
    rm.vis.headlight.ambient[:] = [0.0, 0.0, 0.0]
    rm.vis.headlight.specular[:] = [0.0, 0.0, 0.0]

    # ── 2. 关闭所有场景光源 ──
    if rm.nlight > 0:
        print(f"  ► 将 {rm.nlight} 个场景光源 diffuse/ambient/specular 全部设为 0")
        model.light_diffuse[:] = 0.0
        model.light_ambient[:] = 0.0
        model.light_specular[:] = 0.0

    # ── 3. 移动 agentview 相机 ──
    for i in range(rm.ncam):
        name = id2name(rm, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if "agent" in name.lower():
            print(f"  ► 相机 '{name}': pos[2] += 0.5m, fovy += 40°")
            model.cam_pos[i][2] += 0.5
            model.cam_fovy[i] = min(float(model.cam_fovy[i]) + 40, 160)

    sim.forward()
    print("  ✓ 极端 shift 已应用\n")


def main():
    out_dir = pathlib.Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("创建 LIBERO 环境...")
    env, task_desc, init_states = create_env()
    print(f"  task: {task_desc}\n")

    # ── Phase 1: 原始图像 ────────────────────────────────────────────────────
    print("[Phase 1] reset + set_init_state → 获取原始图像")
    env.reset()
    sim = find_sim(env)
    if sim is None:
        print("FATAL: 找不到 sim，无法继续。")
        return

    print_scene_info(sim)

    obs_orig = env.set_init_state(init_states[0])
    img_original = get_image(obs_orig)

    # ── Phase 2: 应用极端 shift ──────────────────────────────────────────────
    print("\n[Phase 2] 应用极端 domain shift")
    apply_extreme_shift(sim)

    # 重新 set_init_state（不 reset，避免重建 sim）→ 触发 forward + render
    obs_shifted = env.set_init_state(init_states[0])
    img_shifted = get_image(obs_shifted)

    # ── Phase 3: 计算差异 ────────────────────────────────────────────────────
    print("[Phase 3] 计算图像差异")
    diff = np.abs(img_original.astype(np.float32) - img_shifted.astype(np.float32))

    max_diff = diff.max()
    mean_diff = diff.mean()
    pct_changed = (diff > 1.0).mean() * 100
    orig_brightness = img_original.mean()
    shift_brightness = img_shifted.mean()

    print(f"\n{'='*70}")
    print("  诊断结果")
    print(f"{'='*70}")
    print(f"  原始图像: shape={img_original.shape}, mean brightness={orig_brightness:.1f}")
    print(f"  shifted:  shape={img_shifted.shape}, mean brightness={shift_brightness:.1f}")
    print(f"  最大像素差: {max_diff:.1f}")
    print(f"  平均像素差: {mean_diff:.2f}")
    print(f"  变化像素比: {pct_changed:.1f}%")
    print()

    if max_diff < 1.0:
        print("  ⚠️  结论: 图像完全没有变化！")
        print("     Domain shift 完全没有生效。")
        print("     → 检查 sim.model 是否是只读副本")
        print("     → 检查 set_init_state 是否重新渲染了图像")
        print("     → 尝试 reset 后直接 step 获取 obs，而非 set_init_state")
    elif pct_changed < 5.0:
        print("  ⚠️  结论: 图像变化微小。")
        print("     shift 部分生效，但 headlight 可能仍在主导。")
    else:
        print("  ✓  结论: Domain shift 明确生效！")
        print("     现有实验配置值太温和，需要加大扰动幅度。")

    if shift_brightness > 50 and rm_nlight_is_zero(sim):
        print("\n  💡 关键提示: 场景无场景光源 + headlight 关闭后亮度仍 > 50,")
        print("     说明渲染可能有其他光源或 headlight 关闭没生效。")

    # ── 保存图像 ─────────────────────────────────────────────────────────────
    try:
        import imageio
        imageio.imwrite(str(out_dir / "diagnostic_original.png"), img_original)
        imageio.imwrite(str(out_dir / "diagnostic_shifted.png"), img_shifted)
        diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
        imageio.imwrite(str(out_dir / "diagnostic_diff.png"), diff_vis)
        print(f"\n  图像已保存到: {out_dir}/")
    except Exception as e:
        print(f"\n  保存图像失败: {e}")

    env.close()
    print("\n诊断完成。\n")


def rm_nlight_is_zero(sim):
    try:
        return get_raw_model(sim.model).nlight == 0
    except:
        return False


if __name__ == "__main__":
    main()
    