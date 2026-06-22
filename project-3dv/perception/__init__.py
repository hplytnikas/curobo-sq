"""
project_3dv.perception — tabletop RGB-D perception for robot manipulation.

Pipeline stages (standalone functions)
---------------------------------------
    depth_to_pointcloud   Stage 1 — depth image + intrinsics → (N,3) cloud
    remove_table          Stage 2 — RANSAC plane + depth/height/radius filter
    segment_instances     Stage 3 — DBSCAN + multi-pass object splitting
    classify_shape_hint   Stage 4a — geometry heuristic for SQ warm-start
    fit_superquadrics     Stage 4b — SQ fitting per segment → List[MultiSQFit]

Orchestrator
-------------
    TabletopPerception    Runs stages 2–4a; returns PerceptionResult.
                          Call fit_superquadrics() on the result for stage 4b.

Data structures
----------------
    ObjectSegment         One segmented object (points, bbox, shape_type)
    PerceptionResult      Full stages 2–4a output (objects + table geometry)

Superquadric fitting
---------------------
    SuperquadricFitter    Classic Levenberg-Marquardt fitter
    SuperdecFitter        Neural fitter (defaults to fine-tuned tabletop v2)
    SuperquadricFit       Single fitted primitive (11 params + quality)
    MultiSQFit            Set of primitives for one object segment
    fits_to_curobo_obstacles  Export to CuRobo collision-checking format

Scene (CuRobo / planning interface)
-------------------------------------
    Scene                 Wraps Superquadrics; .get_signed_distance() for cuRobo
    sq_fits_to_npz        Export SuperquadricFit list to SuperDec .npz format

Grasp planning
---------------
    GraspSelector         Per-primitive grasp candidates for Franka Panda
    GraspCandidate        Single grasp (pose, gripper_width, score)
"""

from .pipeline import (
    Frame,
    SuperquadricWorld,
    get_world_pointcloud,
    fit_superquadrics_world,
    superquadrics_to_curobo_world,
    MULTIVIEW,
)

from .pipeline import (
    depth_to_pointcloud,
    pointcloud_from_depth,
    remove_table,
    segment_instances,
    segment_instances_dual,
    classify_shape_hint,
    fit_superquadrics,
    TabletopPerception,
    ObjectSegment,
    PerceptionResult,
    PerceptionTimer,
    SQWorldModel,
    single_frame_pipeline,
)
from .superquadric import (
    SuperquadricFitter,
    SuperquadricFit,
    MultiSQFit,
    fits_to_curobo_obstacles,
    sq_type_from_exponents,
)
from .superdec_utils import (
    Scene,
    Superquadrics,
    sq_fits_to_npz,
)
from .grasp_from_sq import (
    GraspSelector,
    GraspCandidate,
)

# SuperdecFitter requires torch + the superdec package; import lazily to avoid
# hard-failing when those are not installed (e.g., in CI without GPU deps).
try:
    from .superdec_fitter import SuperdecFitter
except ImportError:
    SuperdecFitter = None  # type: ignore[assignment,misc]

__all__ = [
    # two-stage pipeline
    "Frame",
    "SuperquadricWorld",
    "get_world_pointcloud",
    "fit_superquadrics_world",
    "superquadrics_to_curobo_world",
    "MULTIVIEW",
    # pipeline stages
    "depth_to_pointcloud",
    "pointcloud_from_depth",
    "remove_table",
    "segment_instances",
    "segment_instances_dual",
    "classify_shape_hint",
    "fit_superquadrics",
    # orchestrator + data structures
    "TabletopPerception",
    "ObjectSegment",
    "PerceptionResult",
    "PerceptionTimer",
    "SQWorldModel",
    # single-frame entry point
    "single_frame_pipeline",
    # SQ fitting
    "SuperquadricFitter",
    "SuperdecFitter",
    "SuperquadricFit",
    "MultiSQFit",
    "fits_to_curobo_obstacles",
    "sq_type_from_exponents",
    # scene / planning interface
    "Scene",
    "Superquadrics",
    "sq_fits_to_npz",
    # grasp planning
    "GraspSelector",
    "GraspCandidate",
]
