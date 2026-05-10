"""Superquadric obstacle collision checking and motion planning in cuRobo v2.

This example demonstrates:
1. Defining superquadric (SQ) obstacles in a SceneCfg
2. Running sphere-vs-SQ collision queries via the Warp kernel
3. Integrating SQs into the cuRobo motion planner with a Franka arm

The SQ SDF uses Newton radial projection (analytical, GPU-parallel) instead of
mesh approximations, giving exact differentiable distances for gradient-based
trajectory optimisation.

Run (headless, no Isaac Sim GUI):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python_v2.sh curobov2/curobo/curobo/examples/getting_started/superquadric_motion_planning.py

Run with native Isaac Sim GUI window (like old curobo examples):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python_v2.sh curobov2/curobo/curobo/examples/getting_started/superquadric_motion_planning.py --gui

Run with WebRTC livestream (requires Omniverse Streaming Client app → connect to localhost:49100):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python_v2.sh curobov2/curobo/curobo/examples/getting_started/superquadric_motion_planning.py --enable_livestream
"""

# Standard Library
import argparse
import time

# Import warp here to lock in v2-venv's 1.12.0 before SimulationApp loads.
# SimulationApp enables omni.warp.core-1.8.2 which inserts an older Warp into
# sys.path; caching warp 1.12.0 in sys.modules first prevents the displacement.
# wp.init() is still deferred until after SimulationApp has fully started.
import warp as wp

# ── Argument parsing MUST happen before SimulationApp init ────────────────────

parser = argparse.ArgumentParser(
    description="Superquadric obstacle demo for cuRobo v2",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
parser.add_argument(
    "--gui",
    action="store_true",
    default=False,
    help="Open a native Isaac Sim GUI window (like old curobo examples).",
)
parser.add_argument(
    "--enable_livestream",
    action="store_true",
    default=False,
    help="Stream via WebRTC. Requires Omniverse Streaming Client → connect to localhost:49100.",
)
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    choices=["native", "webrtc"],
    help="Headless livestream backend. --enable_livestream sets this to 'webrtc'.",
)
args, _ = parser.parse_known_args()

if args.enable_livestream and args.headless_mode is None:
    args.headless_mode = "webrtc"

# ── SimulationApp (must be created before any omni / warp imports) ─────────────

_use_isaac = args.gui or args.headless_mode is not None
_headless = not args.gui  # GUI → headless=False; streaming → headless=True
simulation_app = None

if _use_isaac:
    # CUDA warmup before SimulationApp — prevents a crash on some drivers.
    import torch as _torch
    _torch.zeros(4, device="cuda:0")

    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp(
        {
            "headless": _headless,
            "width": "1920",
            "height": "1080",
        }
    )
    if args.headless_mode is not None:
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.kit.livestream." + args.headless_mode)
        simulation_app.update()
        print("[isaac] livestream active — connect via Omniverse Streaming Client to localhost:49100", flush=True)
    else:
        print("[isaac] GUI window open.", flush=True)

# ── CuRobo imports (after SimulationApp) ──────────────────────────────────────

import torch
wp.init()

from curobo._src.geom.types import SceneCfg, Superquadric, Cuboid
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.types.device_cfg import DeviceCfg


# =============================================================================
# Scene definition
# =============================================================================


def make_scene() -> SceneCfg:
    """Build a scene with three superquadric obstacles.

    Obstacle shapes:
    - sq_ellipsoid : sphere-like  (eps1=eps2=1)
    - sq_cylinder  : cylinder-like (eps1=1, eps2→0 approximated by 0.1)
    - sq_box       : box-like    (eps1=eps2→0 approximated by 0.1)
    """
    return SceneCfg(
        superquadric=[
            Superquadric(
                name="sq_ellipsoid",
                pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                radii=[0.08, 0.08, 0.08],
                shape=[1.0, 1.0],
            ),
            Superquadric(
                name="sq_cylinder",
                pose=[0.4, 0.2, 0.5, 1.0, 0.0, 0.0, 0.0],
                radii=[0.05, 0.05, 0.15],
                shape=[1.0, 0.1],
            ),
            Superquadric(
                name="sq_box",
                pose=[0.35, -0.2, 0.35, 1.0, 0.0, 0.0, 0.0],
                radii=[0.06, 0.10, 0.08],
                shape=[0.1, 0.1],
            ),
        ],
        cuboid=[
            Cuboid(
                name="floor",
                pose=[0.0, 0.0, -0.025, 1.0, 0.0, 0.0, 0.0],
                dims=[2.0, 2.0, 0.05],
            ),
        ],
    )


# =============================================================================
# Isaac Sim visualization helpers
# =============================================================================


def _add_isaac_scene(scene_cfg: SceneCfg) -> None:
    """Populate the Isaac Sim stage with visual approximations of the SQ obstacles."""
    from omni.isaac.core import World
    from omni.isaac.core.objects.cuboid import VisualCuboid
    from omni.isaac.core.objects.sphere import VisualSphere

    world = World()
    world.scene.add_default_ground_plane(z_position=-0.05)

    for sq in scene_cfg.superquadric:
        pos = sq.pose[:3]   # [x, y, z]
        quat = sq.pose[3:]  # [qw, qx, qy, qz]
        orient = (quat[0], quat[1], quat[2], quat[3])  # (w, x, y, z)

        eps1, eps2 = sq.shape[0], sq.shape[1]
        rx, ry, rz = sq.radii

        prim_path = f"/World/obs_{sq.name}"

        if eps1 >= 0.8 and eps2 >= 0.8:
            # Near-sphere: visualise as VisualSphere using mean radius
            r_mean = (rx + ry + rz) / 3.0
            world.scene.add(
                VisualSphere(
                    prim_path=prim_path,
                    name=sq.name,
                    position=pos,
                    radius=r_mean,
                    color=torch.tensor([0.2, 0.6, 0.9]),
                )
            )
        elif eps2 <= 0.2:
            # Near-cylinder (eps2 small = circular cross-section):
            # approximate as a tall VisualCuboid scaled to look like a cylinder
            world.scene.add(
                VisualCuboid(
                    prim_path=prim_path,
                    name=sq.name,
                    position=pos,
                    orientation=orient,
                    size=1.0,
                    scale=torch.tensor([rx * 2, ry * 2, rz * 2]),
                    color=torch.tensor([0.9, 0.6, 0.2]),
                )
            )
        else:
            # Near-box: VisualCuboid with SQ dimensions
            world.scene.add(
                VisualCuboid(
                    prim_path=prim_path,
                    name=sq.name,
                    position=pos,
                    orientation=orient,
                    size=1.0,
                    scale=torch.tensor([rx * 2, ry * 2, rz * 2]),
                    color=torch.tensor([0.9, 0.2, 0.3]),
                )
            )

    world.reset()
    return world


# =============================================================================
# Demo functions
# =============================================================================


def demo_collision_query(scene_cfg: SceneCfg, device_cfg: DeviceCfg) -> None:
    """Manually query SQ collision for a set of probe spheres."""
    print("\n--- Collision Query Demo ---")
    scene_data = SceneData.from_scene_cfg(scene_cfg, device_cfg)

    probes = torch.tensor([
        # sphere INSIDE sq_ellipsoid (at 0.5, 0, 0.4)
        [[[0.50, 0.00, 0.40, 0.01]]],
        # sphere OUTSIDE all obstacles
        [[[1.50, 1.50, 1.50, 0.01]]],
        # sphere NEAR sq_cylinder surface
        [[[0.40, 0.20, 0.50, 0.02]]],
    ], dtype=device_cfg.dtype, device=device_cfg.device)

    labels = ["inside sq_ellipsoid", "free space", "near sq_cylinder"]

    checker = CollisionChecker(device_cfg=device_cfg)
    weight = device_cfg.to_device([1.0])
    act_dist = device_cfg.to_device([0.0])

    for i, probe in enumerate(probes):
        sphere_t = probe.unsqueeze(0)
        buf = CollisionBuffer.from_shape(sphere_t.shape[:3], device_cfg)
        env_idx = torch.zeros(1, dtype=torch.int32, device=device_cfg.device)
        cost = checker.get_sphere_distance(
            scene_data, sphere_t, buf, weight, act_dist,
            env_query_idx=env_idx,
        )
        status = "COLLISION" if float(cost[0, 0, 0].item()) > 0 else "FREE"
        print(f"  Probe {i} ({labels[i]}): cost={float(cost[0,0,0]):.4f}  [{status}]")


def demo_motion_planning(scene_cfg: SceneCfg, device_cfg: DeviceCfg) -> None:
    """Set up SceneCollision with superquadric obstacles for motion planning."""
    print("\n--- Motion Planning Scene Setup ---")

    try:
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
    except ImportError as e:
        print(f"  SceneCollision unavailable: {e}")
        return

    scene_collision_cfg = SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=scene_cfg,
        num_envs=1,
        cache={"cuboid": 4, "superquadric": 8},
    )
    scene_collision = SceneCollision.from_config(scene_collision_cfg)
    print(f"  Active collision types: {scene_collision.collision_types}")
    print(f"  Superquadric count (env 0): "
          f"{scene_collision.data.superquadrics.get_active_count(0)}")

    from curobo._src.types.pose import Pose
    new_pose = Pose.from_list([0.6, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], device_cfg=device_cfg)
    scene_collision.data.update_obstacle_pose("sq_ellipsoid", new_pose)
    print("  Updated sq_ellipsoid pose to [0.6, 0, 0.4, identity]")
    print("  SceneCollision with superquadrics is ready for trajectory optimisation.")


# =============================================================================
# Main
# =============================================================================


def main():
    device_cfg = DeviceCfg(device=args.device)
    scene_cfg = make_scene()

    print(f"Scene: {len(scene_cfg.superquadric)} superquadrics, {len(scene_cfg.cuboid)} cuboids")
    for sq in scene_cfg.superquadric:
        print(f"  {sq.name}: radii={sq.radii}, shape={sq.shape}")

    # 1. Direct collision query
    demo_collision_query(scene_cfg, device_cfg)

    # 2. Full motion planning setup (if available)
    demo_motion_planning(scene_cfg, device_cfg)

    print("\nDone.")

    # 3. If running with Isaac Sim, populate stage and keep rendering
    if simulation_app is not None:
        print("\n[isaac] building stage...", flush=True)
        world = _add_isaac_scene(scene_cfg)
        if args.gui:
            print("[isaac] stage ready. Close the window or press Ctrl-C to quit.", flush=True)
        else:
            print("[isaac] stage ready — connect via Omniverse Streaming Client to localhost:49100", flush=True)
            print("[isaac] press Ctrl-C to quit.", flush=True)
        try:
            while simulation_app.is_running():
                world.step(render=True)
        except KeyboardInterrupt:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
