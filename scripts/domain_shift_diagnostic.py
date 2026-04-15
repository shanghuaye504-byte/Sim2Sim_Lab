#!/usr/bin/env python3
"""
domain_shift_diagnostic.py
==========================
Standalone diagnostic script: verifies whether domain shift actually affects rendered output.

Output:
    diagnostic_output/
    ├── diagnostic_original.png
    ├── diagnostic_shifted.png
    ├── diagnostic_diff.png       (difference x5 amplified)
    └── Terminal output: pixel difference statistics + scene info

Usage:
    python domain_shift_diagnostic.py
"""

import os
import sys
import pathlib

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
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
    """Try multiple paths to find the sim object"""
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
    """Get the underlying mujoco.MjModel (bypassing robosuite wrapper)"""
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
    print("  Scene Info")
    print(f"{'='*70}")
    print(f"  model type     = {type(model).__name__}")
    print(f"  raw_model type = {type(rm).__name__}")
    print(f"  ngeom={rm.ngeom}, nlight={rm.nlight}, ncam={rm.ncam}, nmat={rm.nmat}")

    # ── Headlight (most critical) ──
    print(f"\n  ◆ Headlight:")
    print(f"    active   = {rm.vis.headlight.active}")
    print(f"    diffuse  = {list(rm.vis.headlight.diffuse)}")
    print(f"    ambient  = {list(rm.vis.headlight.ambient)}")
    print(f"    specular = {list(rm.vis.headlight.specular)}")

    # ── Scene lights ──
    print(f"\n  ◆ Scene Lights ({rm.nlight} total):")
    if rm.nlight == 0:
        print("    ⚠️  No scene lights! Illumination relies entirely on headlight.")
        print("    ⚠️  Modifying model.light_* will have no effect.")
    for i in range(rm.nlight):
        name = id2name(rm, mujoco.mjtObj.mjOBJ_LIGHT, i)
        print(f"    light[{i}] = '{name}'")
        print(f"      diffuse  = {list(model.light_diffuse[i])}")
        print(f"      dir      = {list(model.light_dir[i])}")
        print(f"      active   = {model.light_active[i] if hasattr(model, 'light_active') else 'N/A'}")

    # ── Cameras ──
    print(f"\n  ◆ Cameras ({rm.ncam} total):")
    for i in range(rm.ncam):
        name = id2name(rm, mujoco.mjtObj.mjOBJ_CAMERA, i)
        print(f"    cam[{i}] = '{name}'")
        print(f"      pos  = {list(model.cam_pos[i])}")
        print(f"      fovy = {float(model.cam_fovy[i]):.1f}")


def get_image(obs):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def apply_extreme_shift(sim):
    """Apply extreme domain shift — if this shows no difference, it's a code-level issue"""
    model = sim.model
    rm = get_raw_model(model)

    print("\nApplying extreme domain shift:")

    # ── 1. Disable headlight ──
    print("  ► Disabling headlight (active=0, diffuse/ambient/specular=[0,0,0])")
    rm.vis.headlight.active = 0
    rm.vis.headlight.diffuse[:] = [0.0, 0.0, 0.0]
    rm.vis.headlight.ambient[:] = [0.0, 0.0, 0.0]
    rm.vis.headlight.specular[:] = [0.0, 0.0, 0.0]

    # ── 2. Disable all scene lights ──
    if rm.nlight > 0:
        print(f"  ► Setting all {rm.nlight} scene lights diffuse/ambient/specular to 0")
        model.light_diffuse[:] = 0.0
        model.light_ambient[:] = 0.0
        model.light_specular[:] = 0.0

    # ── 3. Move agentview camera ──
    for i in range(rm.ncam):
        name = id2name(rm, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if "agent" in name.lower():
            print(f"  ► Camera '{name}': pos[2] += 0.5m, fovy += 40°")
            model.cam_pos[i][2] += 0.5
            model.cam_fovy[i] = min(float(model.cam_fovy[i]) + 40, 160)

    sim.forward()
    print("  ✓ Extreme shift applied\n")


def main():
    out_dir = pathlib.Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Creating LIBERO environment...")
    env, task_desc, init_states = create_env()
    print(f"  task: {task_desc}\n")

    # ── Phase 1: Original image ──────────────────────────────────────────────
    print("[Phase 1] reset + set_init_state → capture original image")
    env.reset()
    sim = find_sim(env)
    if sim is None:
        print("FATAL: Cannot find sim, unable to continue.")
        return

    print_scene_info(sim)

    obs_orig = env.set_init_state(init_states[0])
    img_original = get_image(obs_orig)

    # ── Phase 2: Apply extreme shift ─────────────────────────────────────────
    print("\n[Phase 2] Applying extreme domain shift")
    apply_extreme_shift(sim)

    # Re-run set_init_state (no reset to avoid rebuilding sim) -> triggers forward + render
    obs_shifted = env.set_init_state(init_states[0])
    img_shifted = get_image(obs_shifted)

    # ── Phase 3: Compute difference ──────────────────────────────────────────
    print("[Phase 3] Computing image difference")
    diff = np.abs(img_original.astype(np.float32) - img_shifted.astype(np.float32))

    max_diff = diff.max()
    mean_diff = diff.mean()
    pct_changed = (diff > 1.0).mean() * 100
    orig_brightness = img_original.mean()
    shift_brightness = img_shifted.mean()

    print(f"\n{'='*70}")
    print("  Diagnostic Results")
    print(f"{'='*70}")
    print(f"  Original image: shape={img_original.shape}, mean brightness={orig_brightness:.1f}")
    print(f"  Shifted:        shape={img_shifted.shape}, mean brightness={shift_brightness:.1f}")
    print(f"  Max pixel diff: {max_diff:.1f}")
    print(f"  Mean pixel diff: {mean_diff:.2f}")
    print(f"  Changed pixel ratio: {pct_changed:.1f}%")
    print()

    if max_diff < 1.0:
        print("  ⚠️  Conclusion: Image is completely unchanged!")
        print("     Domain shift has no effect at all.")
        print("     → Check if sim.model is a read-only copy")
        print("     → Check if set_init_state re-renders the image")
        print("     → Try getting obs via step after reset instead of set_init_state")
    elif pct_changed < 5.0:
        print("  ⚠️  Conclusion: Image change is minimal.")
        print("     Shift partially works, but headlight may still dominate.")
    else:
        print("  ✓  Conclusion: Domain shift is clearly effective!")
        print("     Current experiment config values are too mild; increase perturbation magnitude.")

    if shift_brightness > 50 and rm_nlight_is_zero(sim):
        print("\n  💡 Key hint: No scene lights + headlight disabled but brightness > 50,")
        print("     suggests there are other light sources or headlight disable did not take effect.")

    # ── Save images ──────────────────────────────────────────────────────────
    try:
        import imageio
        imageio.imwrite(str(out_dir / "diagnostic_original.png"), img_original)
        imageio.imwrite(str(out_dir / "diagnostic_shifted.png"), img_shifted)
        diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
        imageio.imwrite(str(out_dir / "diagnostic_diff.png"), diff_vis)
        print(f"\n  Images saved to: {out_dir}/")
    except Exception as e:
        print(f"\n  Failed to save images: {e}")

    env.close()
    print("\nDiagnostic complete.\n")


def rm_nlight_is_zero(sim):
    try:
        return get_raw_model(sim.model).nlight == 0
    except:
        return False


if __name__ == "__main__":
    main()
