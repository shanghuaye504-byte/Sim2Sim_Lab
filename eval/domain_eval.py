#!/usr/bin/env python3
"""
Domain Shift Wrapper for LIBERO Evaluation
===========================================

Core Design:
  1. Monkey-patch main._get_libero_env to inject DomainShiftEnvWrapper
  2. The wrapper automatically re-applies domain shift after each env.reset()
     (because robosuite hard_reset=True rebuilds sim on every reset, losing all model parameters)
  3. set_init_state() does not rebuild sim; it only sets state and re-renders.
     Domain shift remains effective after reset until the next reset.

Supported Domain Shift Types:
  A. Lighting  — intensity, direction, warm/cool color, shadow, active
  B. Camera    — position offset, euler rotation, fovy
  C. Friction  — global, per-geom keyword
  D. Material  — specular, shininess, reflectance (global + per-material keyword)
  E. Geom RGBA — per-geom keyword (preserving existing functionality)

Usage:
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
# Step 1: Add main.py's directory to sys.path and import libero_main
# ═══════════════════════════════════════════════════════════════════════════════
LIBERO_MAIN_DIR = os.environ.get(
    "LIBERO_MAIN_DIR",
    "/app/third_party/openpi/examples/libero",
)
sys.path.insert(0, LIBERO_MAIN_DIR)

import main as libero_main  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Extend Args with a log_dir field (bash script passes --args.log-dir)
#         The original main.py Args lacks this field; without extending, tyro would error
# ═══════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class DomainEvalArgs(libero_main.Args):
    log_dir: str = "data/libero/logs"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Load YAML domain config
# ═══════════════════════════════════════════════════════════════════════════════
def _load_domain_config() -> dict:
    config_file = os.environ.get("DOMAIN_CONFIG_FILE", "").strip()
    if not config_file:
        log.info("[domain_eval] DOMAIN_CONFIG_FILE not set, running with source domain.")
        return {}

    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f) or {}
    log.info(f"[domain_eval] Loaded domain config: {config_file}")

    # ── Added: if YAML has multi-level format, extract by DOMAIN_LEVEL ────────
    level = os.environ.get("DOMAIN_LEVEL", "").strip()
    if "levels" in cfg and level:
        levels = cfg["levels"]
        if level not in levels:
            raise ValueError(
                f"DOMAIN_LEVEL='{level}' does not exist in {config_file}. "
                f"Available levels: {list(levels.keys())}"
            )
        cfg = levels[level]
        log.info(f"[domain_eval] Using level='{level}'")
    elif "levels" in cfg and not level:
        raise ValueError(
            f"{config_file} has multi-level format but DOMAIN_LEVEL env variable is not set. "
            f"Available levels: {list(cfg['levels'].keys())}"
        )
    # If there is no "levels" key, it is a flat format — use as-is (backward compatible)
    # ─────────────────────────────────────────────────────────────────────────

    log.info(f"[domain_eval] Config contents:\n{yaml.dump(cfg, default_flow_style=False)}")
    return cfg


DOMAIN_CONFIG = _load_domain_config()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: MuJoCo Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _find_sim(env):
    """
    Find the robosuite sim object through LIBERO's multi-layer wrappers.
    OffScreenRenderEnv -> ControlEnv.env -> robosuite env.sim
    ControlEnv itself has a @property sim that proxies to self.env.sim
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
    log.warning("[domain_eval] Failed to find sim object! Please check LIBERO/robosuite version.")
    return None


def _get_raw_model(model):
    """
    robosuite's MjSim.model wraps mujoco.MjModel;
    mujoco C APIs (e.g. mj_name2id) need the raw model object.
    """
    return getattr(model, "_model", model)


def _id2name(model, type_enum, idx):
    """Get MuJoCo object name by ID (compatible with robosuite wrapper)"""
    import mujoco
    try:
        return mujoco.mj_id2name(_get_raw_model(model), type_enum, idx) or ""
    except Exception:
        return ""


def _name2id(model, type_enum, name):
    """Get MuJoCo object ID by name (compatible with robosuite wrapper)"""
    import mujoco
    try:
        # Prefer robosuite wrapper methods (e.g. camera_name2id for camera)
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
    # Fallback: call mujoco C API directly
    try:
        result = mujoco.mj_name2id(_get_raw_model(model), type_enum, name)
        return result if result >= 0 else -1
    except Exception:
        return -1


def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    """
    Euler angles (degrees) -> MuJoCo quaternion [w, x, y, z]
    Uses ZYX intrinsic rotation convention: rotate around X(roll) -> Y(pitch) -> Z(yaw)
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
    """Quaternion multiplication q1 x q2, both in [w, x, y, z] format"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Core Domain Shift Application Function
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_domain_shift(env, config: dict, verbose: bool = False) -> None:
    """
    Modify MuJoCo model attributes after env.reset() completes.

    Args:
        env:     OffScreenRenderEnv instance (ControlEnv subclass)
        config:  Domain shift configuration dict loaded from YAML
        verbose: True on first call to print detailed matching info

    Why re-apply after every reset?
        robosuite defaults to hard_reset=True; each env.reset() executes
        _load_model() + _initialize_sim(), completely rebuilding the MjSim object,
        so all previous model modifications are lost.
    """
    import mujoco

    if not config:
        return

    sim = _find_sim(env)
    if sim is None:
        return

    model = sim.model

    # ── If verbose, print scene overview ──────────────────────────────────────
    if verbose:
        log.info(
            f"[domain_eval] Scene overview: "
            f"ngeom={model.ngeom}, nlight={model.nlight}, "
            f"ncam={model.ncam}, nmat={model.nmat}"
        )
        # Print all camera names for user verification
        for i in range(model.ncam):
            cname = _id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            log.info(f"  camera[{i}] = '{cname}', "
                     f"pos={model.cam_pos[i]}, fovy={model.cam_fovy[i]:.1f}")
        # Print all light source names
        for i in range(model.nlight):
            lname = _id2name(model, mujoco.mjtObj.mjOBJ_LIGHT, i)
            log.info(f"  light[{i}] = '{lname}', "
                     f"diffuse={model.light_diffuse[i]}, "
                     f"dir={model.light_dir[i]}")

    # ══════════════════════════════════════════════════════════════════════════
    # A. Lighting Adjustment
    # ══════════════════════════════════════════════════════════════════════════
    lighting = config.get("lighting", {})
    if lighting and model.nlight > 0:
        if verbose:
            log.info(f"[domain_eval][A] Applying lighting adjustment ({model.nlight} light sources): {lighting}")

        # A1. Diffuse intensity scale (controls overall brightness)
        if "diffuse_scale" in lighting:
            s = float(lighting["diffuse_scale"])
            model.light_diffuse[:] = np.clip(model.light_diffuse * s, 0.0, 1.0)

        # A2. Specular intensity scale (controls highlight strength)
        if "specular_scale" in lighting:
            s = float(lighting["specular_scale"])
            model.light_specular[:] = np.clip(model.light_specular * s, 0.0, 1.0)

        # A3. Ambient intensity scale (controls minimum brightness in shadow areas)
        if "ambient_scale" in lighting:
            s = float(lighting["ambient_scale"])
            model.light_ambient[:] = np.clip(model.light_ambient * s, 0.0, 1.0)

        # A4. Light direction offset (changes light incidence angle)
        #     Re-normalize to unit vector after offset
        if "direction_offset" in lighting:
            offset = np.array(lighting["direction_offset"], dtype=np.float64)
            for i in range(model.nlight):
                d = model.light_dir[i] + offset
                norm = np.linalg.norm(d)
                if norm > 1e-8:
                    model.light_dir[i] = d / norm

        # A5. Light position offset
        if "position_offset" in lighting:
            offset = np.array(lighting["position_offset"], dtype=np.float64)
            model.light_pos[:] = model.light_pos + offset

        # A6. Color temperature shift: positive R / negative B = warm; negative R / positive B = cool
        #     Added to diffuse, affecting direct light color
        if "color_shift" in lighting:
            shift = np.array(lighting["color_shift"], dtype=np.float64)
            model.light_diffuse[:] = np.clip(model.light_diffuse + shift, 0.0, 1.0)

        # A7. Shadow toggle (0 = no shadow casting, 1 = shadow casting)
        if "castshadow" in lighting:
            model.light_castshadow[:] = int(lighting["castshadow"])

        # A8. Light activation toggle (0 = off, 1 = on)
        #     Note: disabling all lights will result in a completely dark scene
        if "active" in lighting:
            model.light_active[:] = int(lighting["active"])

    # ══════════════════════════════════════════════════════════════════════════
    # B. Camera Adjustment
    # ══════════════════════════════════════════════════════════════════════════
    camera = config.get("camera", {})
    cam_shifts = camera.get("shifts", [])
    if cam_shifts and model.ncam > 0:
        if verbose:
            log.info(f"[domain_eval][B] Applying camera adjustment ({len(cam_shifts)} shift entries)")

        for shift in cam_shifts:
            cam_name = shift.get("name", "")
            cam_id = _name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                log.warning(f"[domain_eval][B] Camera '{cam_name}' not found, skipping. "
                            f"Available cameras: {[_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]}")
                continue

            # B1. Position offset [dx, dy, dz] (meters)
            #     agentview and other fixed cameras: world coordinate offset
            #     eye_in_hand and other attached cameras: offset relative to parent body
            if "pos_offset" in shift:
                offset = np.array(shift["pos_offset"], dtype=np.float64)
                model.cam_pos[cam_id] += offset
                if verbose:
                    log.info(f"  cam '{cam_name}': pos += {offset} → {model.cam_pos[cam_id]}")

            # B2. Euler angle offset [roll, pitch, yaw] (degrees)
            #     Left-multiply the offset quaternion onto the original quaternion
            if "euler_offset_deg" in shift:
                euler = shift["euler_offset_deg"]
                dq = _euler_deg_to_quat(euler[0], euler[1], euler[2])
                orig_q = model.cam_quat[cam_id].copy()
                new_q = _quat_mul(dq, orig_q)
                new_q /= np.linalg.norm(new_q)  # Ensure unit quaternion
                model.cam_quat[cam_id] = new_q
                if verbose:
                    log.info(f"  cam '{cam_name}': euler offset {euler}° applied")

            # B3. Field of view offset (degrees), clipped to [10, 160]
            if "fovy_offset" in shift:
                fov_offset = float(shift["fovy_offset"])
                old_fov = float(model.cam_fovy[cam_id])
                model.cam_fovy[cam_id] = np.clip(old_fov + fov_offset, 10.0, 160.0)
                if verbose:
                    log.info(f"  cam '{cam_name}': fovy {old_fov:.1f} → {float(model.cam_fovy[cam_id]):.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    # C. Friction Adjustment
    #    geom_friction is an (ngeom, 3) array: [sliding, torsional, rolling]
    # ══════════════════════════════════════════════════════════════════════════
    friction = config.get("friction", {})
    if friction and model.ngeom > 0:
        if verbose:
            log.info(f"[domain_eval][C] Applying friction adjustment ({model.ngeom} geoms)")

        # C1. Global friction scale (scalar -> uniform scaling for all 3 components; list -> per-component scaling)
        if "global_scale" in friction:
            gs = friction["global_scale"]
            if isinstance(gs, (int, float)):
                gs = [float(gs)] * 3
            gs = np.array(gs, dtype=np.float64)
            model.geom_friction[:] = np.maximum(model.geom_friction * gs, 0.0)
            if verbose:
                log.info(f"  Global friction scale: x{gs}")

        # C2. Scale friction for specific geoms by name keyword
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
                log.info(f"  Friction keyword '{keyword}': matched {matched} geoms, scale={fs}")

    # ══════════════════════════════════════════════════════════════════════════
    # D. Material Optical Properties Adjustment
    # ══════════════════════════════════════════════════════════════════════════
    material_cfg = config.get("material", {})
    if material_cfg and model.nmat > 0:
        if verbose:
            log.info(f"[domain_eval][D] Applying material optical adjustment ({model.nmat} materials)")

        # D1. Global specular scale
        if "global_specular_scale" in material_cfg:
            s = float(material_cfg["global_specular_scale"])
            model.mat_specular[:] = np.clip(model.mat_specular * s, 0.0, 1.0)
            if verbose:
                log.info(f"  Global specular x{s}")

        # D2. Global shininess scale
        if "global_shininess_scale" in material_cfg:
            s = float(material_cfg["global_shininess_scale"])
            model.mat_shininess[:] = np.clip(model.mat_shininess * s, 0.0, 1.0)
            if verbose:
                log.info(f"  Global shininess x{s}")

        # D3. Global reflectance scale
        if "global_reflectance_scale" in material_cfg:
            s = float(material_cfg["global_reflectance_scale"])
            model.mat_reflectance[:] = np.clip(model.mat_reflectance * s, 0.0, 1.0)
            if verbose:
                log.info(f"  Global reflectance x{s}")

        # D4. Set specific material properties by name keyword (absolute values, not scaling)
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
                log.info(f"  Material keyword '{keyword}': matched {matched} materials")

    # ══════════════════════════════════════════════════════════════════════════
    # E. Geom RGBA Adjustment (preserving existing functionality, unified into new framework)
    # ══════════════════════════════════════════════════════════════════════════
    geom_shifts = config.get("geom_rgba_shifts", [])
    if geom_shifts and model.ngeom > 0:
        if verbose:
            log.info(f"[domain_eval][E] Applying geom RGBA adjustment")
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
                log.info(f"  geom_rgba keyword '{keyword}': matched {matched} geoms")

    # ── Call forward() to propagate physics parameter changes (e.g. friction) to derived quantities ──
    # Visual parameters (lighting/camera/material) take effect automatically on the next render,
    # but forward() updates contact-related derived quantities, which helps with friction changes
    sim.forward()

    if verbose:
        log.info("[domain_eval] ✓ Domain shift applied successfully.")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Environment Wrapper — Solves the core issue of hard_reset losing modifications
# ═══════════════════════════════════════════════════════════════════════════════

class DomainShiftEnvWrapper:
    """
    Wraps OffScreenRenderEnv to automatically re-apply domain shift after each reset().

    Why is this wrapper needed?
    ───────────────────────────
    robosuite defaults to hard_reset=True; each env.reset() will:
      1. _load_model()      -> rebuild the model from XML
      2. _initialize_sim()  -> create a brand new MjSim object
    This means all previous modifications to model parameters (lighting, camera, friction, etc.)
    are completely lost.

    Since the evaluation loop calls reset() for each episode, we must re-apply after every reset.

    Call order guarantee:
    ─────────────────────
    The flow in eval_libero:
      env.reset()                              <- rebuilds sim; we inject shift right after
      obs = env.set_init_state(init_state)     <- sets qpos/qvel, calls forward() + render
                                                  domain shift is already in effect at this point ✓
      for step:
        obs, r, d, i = env.step(action)        <- physics step + render, shift persists ✓
    """

    def __init__(self, env, config: dict):
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_reset_count", 0)

    def reset(self):
        """Call original reset(), then re-apply domain shift."""
        obs = self._env.reset()

        # verbose=True on first reset to print detailed scene info for debugging
        verbose = (self._reset_count == 0)
        _apply_domain_shift(self._env, self._config, verbose=verbose)
        object.__setattr__(self, "_reset_count", self._reset_count + 1)

        return obs

    def set_init_state(self, init_state):
        """
        set_init_state does not rebuild sim (only sets qpos/qvel + forward + render),
        so domain shift applied in the previous reset is still in effect here.
        The returned obs already includes domain shift effects.
        """
        return self._env.set_init_state(init_state)

    def step(self, action):
        """Directly forward; domain shift persists within this episode."""
        return self._env.step(action)

    def seed(self, s):
        return self._env.seed(s)

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        """Forward all attributes/methods not explicitly defined to the underlying env."""
        return getattr(self._env, name)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7: Monkey-patch _get_libero_env
# ═══════════════════════════════════════════════════════════════════════════════

_original_get_libero_env = libero_main._get_libero_env


def _patched_get_libero_env(task, resolution, seed):
    """
    Call the original function to create env, then wrap with DomainShiftEnvWrapper.

    Note: env has not been reset() yet at this point; domain shift will be
    automatically applied when eval_libero() first calls env.reset().
    """
    env, task_description = _original_get_libero_env(task, resolution, seed)
    if DOMAIN_CONFIG:
        env = DomainShiftEnvWrapper(env, DOMAIN_CONFIG)
        log.info(f"[domain_eval] Environment wrapped with DomainShiftEnvWrapper "
                 f"(task: {task_description})")
    return env, task_description


libero_main._get_libero_env = _patched_get_libero_env
log.info("[domain_eval] ✓ Successfully patched _get_libero_env")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8: Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def _domain_eval_main(args: DomainEvalArgs) -> None:
    """
    Thin wrapper: extends Args to support the log_dir field, then delegates to eval_libero.
    DomainEvalArgs inherits from Args, so all original fields are fully compatible.
    """
    log.info(f"[domain_eval] log_dir = {args.log_dir}")
    log.info(f"[domain_eval] video_out_path = {args.video_out_path}")
    libero_main.eval_libero(args)


if __name__ == "__main__":
    import tyro
    tyro.cli(_domain_eval_main) 