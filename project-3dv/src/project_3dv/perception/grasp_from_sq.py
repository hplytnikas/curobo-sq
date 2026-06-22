"""
grasp_from_sq.py
================
Converts SuperDec MultiSQFit primitive decompositions into ranked grasp
candidates for the Franka Panda gripper.

Each detected object segment produces a GraspCandidate with:
  - position       (3,)  — grasp point in world frame
  - orientation    (3,3) — rotation matrix: z-axis = approach direction
  - gripper_width  float — metres, clamped to Franka max (0.08 m)
  - score          float — higher is better
  - primitive_type str   — 'Cylinder' | 'Cuboid' | 'Ellipsoid' | 'Other'
  - primitive_idx  int   — index into MultiSQFit.primitives

Usage
-----
    from grasp_from_sq import GraspSelector
    from project_3dv.perception.pipeline import TabletopPerception

    perception = TabletopPerception(...)
    selector   = GraspSelector(robot_base=np.array([0, 0, 0]))

    result = perception.run(points, rgb)
    for obj in result.objects:
        grasp = selector.best_grasp(obj.sq_fit)
        if grasp is not None:
            print(grasp)

CuRobo integration
------------------
    # convert to CuRobo Pose
    from curobo.types.math import Pose
    import torch

    pos  = torch.tensor(grasp.position,    dtype=torch.float32)
    quat = grasp.quaternion()              # (w, x, y, z)
    pose = Pose(position=pos, quaternion=torch.tensor(quat))
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Franka Panda constants
# ---------------------------------------------------------------------------
FRANKA_MAX_GRIPPER_WIDTH = 0.08   # metres
FRANKA_MIN_GRIPPER_WIDTH = 0.002  # metres
FRANKA_FINGER_LENGTH     = 0.058  # metres — standoff from palm to fingertip


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GraspCandidate:
    """A single grasp candidate for Franka Panda."""
    position:      np.ndarray          # (3,) world frame
    orientation:   np.ndarray          # (3,3) rotation matrix
    gripper_width: float               # metres
    score:         float               # higher = better
    primitive_type: str = 'Other'
    primitive_idx:  int = 0
    object_id:      int = -1

    # approach direction = orientation[:, 2] (z-column)
    @property
    def approach(self) -> np.ndarray:
        return self.orientation[:, 2]

    def quaternion(self) -> np.ndarray:
        """Return (w, x, y, z) quaternion from rotation matrix."""
        R = self.orientation
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z], dtype=np.float32)

    def __repr__(self):
        p = self.position
        return (f"GraspCandidate(type={self.primitive_type}, "
                f"pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}), "
                f"width={self.gripper_width*100:.1f}cm, "
                f"score={self.score:.3f})")


@dataclass
class GraspSet:
    """All grasp candidates for one object segment."""
    object_id:  int
    candidates: List[GraspCandidate] = field(default_factory=list)

    @property
    def best(self) -> Optional[GraspCandidate]:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda g: g.score)

    def top_k(self, k: int = 3) -> List[GraspCandidate]:
        return sorted(self.candidates, key=lambda g: g.score, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Helper: build rotation matrix from approach direction
# ---------------------------------------------------------------------------
def _rotmat_from_approach(approach: np.ndarray,
                           up_hint: np.ndarray = np.array([0., 0., 1.])) -> np.ndarray:
    """
    Build a rotation matrix where z = approach direction.
    x and y are computed to form a right-handed frame.
    """
    z = approach / (np.linalg.norm(approach) + 1e-8)
    # pick x perpendicular to z
    if abs(np.dot(z, up_hint)) > 0.9:
        up_hint = np.array([1., 0., 0.])
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)   # columns = x, y, z axes
    return R


def _euler_to_rotmat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic XYZ Euler angles → rotation matrix."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Per-primitive grasp generators
# ---------------------------------------------------------------------------
def _grasp_cylinder(prim, obj_id: int, prim_idx: int,
                    robot_base: np.ndarray) -> List[GraspCandidate]:
    """
    Cylinder: grasp around the barrel from the side.
    Approach direction = perpendicular to cylinder axis,
    pointing from robot base toward object.
    Gripper width = 2 * min(sx, sy) with small clearance.
    """
    R_obj  = _euler_to_rotmat(prim.rx, prim.ry, prim.rz)
    center = np.array([prim.tx, prim.ty, prim.tz])
    # cylinder axis = z column of object rotation
    axis   = R_obj[:, 2]

    # diameter in cross-section
    diameter = 2.0 * min(prim.sx, prim.sy)
    width    = np.clip(diameter + 0.002, FRANKA_MIN_GRIPPER_WIDTH,
                       FRANKA_MAX_GRIPPER_WIDTH)

    # approach from robot side, perpendicular to axis
    to_obj   = center - robot_base
    approach = to_obj - np.dot(to_obj, axis) * axis
    norm     = np.linalg.norm(approach)
    if norm < 1e-3:
        approach = np.array([1., 0., 0.])
    else:
        approach /= norm

    # grasp point: on cylinder surface, offset by finger length
    grasp_pos = center - approach * (prim.sx + FRANKA_FINGER_LENGTH)
    R_grasp   = _rotmat_from_approach(approach)

    # score: prefer fatter cylinders (easier to grasp), penalise far objects
    dist  = np.linalg.norm(center - robot_base)
    score = diameter * 10.0 - dist * 0.1 + float(prim.shape_conf)

    return [GraspCandidate(
        position=grasp_pos, orientation=R_grasp,
        gripper_width=width, score=score,
        primitive_type='Cylinder', primitive_idx=prim_idx,
        object_id=obj_id,
    )]


def _grasp_cuboid(prim, obj_id: int, prim_idx: int,
                  robot_base: np.ndarray) -> List[GraspCandidate]:
    """
    Cuboid: grasp from the two narrowest faces.
    Generates one candidate per graspable face pair.
    """
    R_obj  = _euler_to_rotmat(prim.rx, prim.ry, prim.rz)
    center = np.array([prim.tx, prim.ty, prim.tz])
    extents = np.array([prim.sx, prim.sy, prim.sz])   # half-extents

    candidates = []
    # try all 3 axis pairs as approach directions
    for axis_idx in range(3):
        width_idx = [i for i in range(3) if i != axis_idx]
        width = 2.0 * min(extents[width_idx[0]], extents[width_idx[1]])
        if width > FRANKA_MAX_GRIPPER_WIDTH:
            continue   # too wide to grasp

        approach = R_obj[:, axis_idx]
        # pick direction toward robot
        to_obj = center - robot_base
        if np.dot(approach, to_obj) < 0:
            approach = -approach

        depth    = extents[axis_idx]
        grasp_pos = center - approach * (depth + FRANKA_FINGER_LENGTH)
        R_grasp  = _rotmat_from_approach(approach)

        dist  = np.linalg.norm(center - robot_base)
        score = (FRANKA_MAX_GRIPPER_WIDTH - width) * 5.0 - dist * 0.1 + float(prim.shape_conf)

        candidates.append(GraspCandidate(
            position=grasp_pos, orientation=R_grasp,
            gripper_width=np.clip(width + 0.002, FRANKA_MIN_GRIPPER_WIDTH,
                                  FRANKA_MAX_GRIPPER_WIDTH),
            score=score,
            primitive_type='Cuboid', primitive_idx=prim_idx,
            object_id=obj_id,
        ))

    return candidates


def _grasp_ellipsoid(prim, obj_id: int, prim_idx: int,
                     robot_base: np.ndarray) -> List[GraspCandidate]:
    """
    Ellipsoid: approach along the shortest axis (most pinchable direction).
    """
    center  = np.array([prim.tx, prim.ty, prim.tz])
    R_obj   = _euler_to_rotmat(prim.rx, prim.ry, prim.rz)
    extents = np.array([prim.sx, prim.sy, prim.sz])

    short_axis_idx = int(np.argmin(extents))
    approach = R_obj[:, short_axis_idx]
    to_obj   = center - robot_base
    if np.dot(approach, to_obj) < 0:
        approach = -approach

    width = 2.0 * min(extents[i] for i in range(3) if i != short_axis_idx)
    width = np.clip(width + 0.002, FRANKA_MIN_GRIPPER_WIDTH, FRANKA_MAX_GRIPPER_WIDTH)

    grasp_pos = center - approach * (extents[short_axis_idx] + FRANKA_FINGER_LENGTH)
    R_grasp   = _rotmat_from_approach(approach)

    dist  = np.linalg.norm(center - robot_base)
    score = float(prim.shape_conf) - dist * 0.1

    return [GraspCandidate(
        position=grasp_pos, orientation=R_grasp,
        gripper_width=width, score=score,
        primitive_type='Ellipsoid', primitive_idx=prim_idx,
        object_id=obj_id,
    )]


def _grasp_other(prim, obj_id: int, prim_idx: int,
                 robot_base: np.ndarray) -> List[GraspCandidate]:
    """Fallback: top-down grasp on centroid."""
    center    = np.array([prim.tx, prim.ty, prim.tz])
    approach  = np.array([0., 0., -1.])   # straight down
    width     = np.clip(2.0 * min(prim.sx, prim.sy) + 0.005,
                        FRANKA_MIN_GRIPPER_WIDTH, FRANKA_MAX_GRIPPER_WIDTH)
    grasp_pos = center + np.array([0., 0., max(prim.sz, 0.02) + FRANKA_FINGER_LENGTH])
    R_grasp   = _rotmat_from_approach(approach)
    dist      = np.linalg.norm(center - robot_base)
    score     = 0.1 - dist * 0.1

    return [GraspCandidate(
        position=grasp_pos, orientation=R_grasp,
        gripper_width=width, score=score,
        primitive_type='Other', primitive_idx=prim_idx,
        object_id=obj_id,
    )]


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------
class GraspSelector:
    """
    Converts SuperDec MultiSQFit decompositions into ranked grasp candidates.

    Parameters
    ----------
    robot_base : (3,) array
        Position of the robot base in world frame.
        Used to rank grasps by reachability and set approach directions.
    table_height : float
        Z height of table surface in world frame (metres).
        Used to filter grasps that would collide with the table.
    min_score : float
        Candidates below this score are discarded.
    """

    def __init__(self,
                 robot_base: np.ndarray = np.zeros(3),
                 table_height: float = 0.0,
                 min_score: float = -np.inf):
        self.robot_base   = np.asarray(robot_base, dtype=np.float64)
        self.table_height = table_height
        self.min_score    = min_score

    def grasp_candidates(self, sq_fit, object_id: int = 0) -> GraspSet:
        """
        Generate all grasp candidates for one object's MultiSQFit.

        Parameters
        ----------
        sq_fit : MultiSQFit
            Output of SuperdecFitter.fit_adaptive() or fit_batch()[i].
        object_id : int
            Index of the object in the scene (for bookkeeping).

        Returns
        -------
        GraspSet with ranked candidates.
        """
        gset = GraspSet(object_id=object_id)

        for p_idx, prim in enumerate(sq_fit.primitives):
            ptype = getattr(prim, 'shape_type', 'Other')

            if ptype == 'Cylinder':
                cands = _grasp_cylinder(prim, object_id, p_idx, self.robot_base)
            elif ptype == 'Cuboid':
                cands = _grasp_cuboid(prim, object_id, p_idx, self.robot_base)
            elif ptype == 'Ellipsoid':
                cands = _grasp_ellipsoid(prim, object_id, p_idx, self.robot_base)
            else:
                cands = _grasp_other(prim, object_id, p_idx, self.robot_base)

            for c in cands:
                # filter: don't grasp below table
                if c.position[2] < self.table_height + 0.01:
                    continue
                # filter: min score
                if c.score < self.min_score:
                    continue
                gset.candidates.append(c)

        return gset

    def best_grasp(self, sq_fit, object_id: int = 0) -> Optional[GraspCandidate]:
        """Return the single best grasp candidate for one object."""
        return self.grasp_candidates(sq_fit, object_id).best

    def plan_clearing_sequence(self, sq_fits: list,
                                table_height: float = 0.0) -> List[GraspCandidate]:
        """
        Given a list of MultiSQFit (one per detected object),
        return an ordered list of best grasps for tabletop clearing.

        Objects are ordered nearest-first (easier for the robot arm).

        Parameters
        ----------
        sq_fits : list of MultiSQFit
        table_height : float

        Returns
        -------
        List of GraspCandidate, one per object (None entries excluded).
        """
        self.table_height = table_height
        grasps = []
        for obj_id, sq_fit in enumerate(sq_fits):
            g = self.best_grasp(sq_fit, obj_id)
            if g is not None:
                grasps.append(g)

        # sort by distance to robot base (nearest first)
        grasps.sort(key=lambda g: np.linalg.norm(g.position - self.robot_base))
        return grasps


# ---------------------------------------------------------------------------
# CuRobo conversion utilities
# ---------------------------------------------------------------------------
def grasp_to_curobo(grasp: GraspCandidate):
    """
    Convert a GraspCandidate to a CuRobo Pose.
    Requires: pip install curobo

    Returns
    -------
    curobo.types.math.Pose
    """
    try:
        import torch
        from curobo.types.math import Pose
    except ImportError:
        raise ImportError("CuRobo not installed. pip install curobo")

    pos  = torch.tensor(grasp.position,    dtype=torch.float32).unsqueeze(0)
    quat = torch.tensor(grasp.quaternion(), dtype=torch.float32).unsqueeze(0)
    return Pose(position=pos, quaternion=quat)


def grasps_to_curobo_batch(grasps: List[GraspCandidate]):
    """
    Convert a list of GraspCandidates to a batched CuRobo Pose.
    Useful for planning multiple grasps in one CuRobo call.
    """
    try:
        import torch
        from curobo.types.math import Pose
    except ImportError:
        raise ImportError("CuRobo not installed.")

    positions = torch.tensor(
        np.stack([g.position for g in grasps]), dtype=torch.float32)
    quats = torch.tensor(
        np.stack([g.quaternion() for g in grasps]), dtype=torch.float32)
    return Pose(position=positions, quaternion=quats)


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    """
    Minimal smoke test using a synthetic MultiSQFit.
    Run from project-3dv root:
        python3 src/project_3dv/perception/grasp_from_sq.py
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    # mock a SuperquadricFit for testing without running the full pipeline
    from dataclasses import dataclass as dc

    @dc
    class MockPrim:
        sx: float; sy: float; sz: float
        e1: float; e2: float
        tx: float; ty: float; tz: float
        rx: float; ry: float; rz: float
        shape_type: str; shape_conf: float
        converged: bool; chamfer_l2: float

    @dc
    class MockMultiSQFit:
        primitives: list

    # simulate a mug: cylinder body
    mug = MockMultiSQFit(primitives=[
        MockPrim(sx=0.04, sy=0.04, sz=0.07,
                 e1=0.2, e2=1.0,
                 tx=0.3, ty=0.1, tz=0.07,
                 rx=0.0, ry=0.0, rz=0.0,
                 shape_type='Cylinder', shape_conf=0.95,
                 converged=True, chamfer_l2=0.002),
    ])

    # simulate a box: cuboid
    box = MockMultiSQFit(primitives=[
        MockPrim(sx=0.06, sy=0.04, sz=0.05,
                 e1=0.2, e2=0.2,
                 tx=0.4, ty=-0.1, tz=0.05,
                 rx=0.0, ry=0.0, rz=0.3,
                 shape_type='Cuboid', shape_conf=0.88,
                 converged=True, chamfer_l2=0.003),
    ])

    selector = GraspSelector(
        robot_base=np.array([0.0, 0.0, 0.0]),
        table_height=0.0,
    )

    print("=== Mug grasps ===")
    gset = selector.grasp_candidates(mug, object_id=0)
    for g in gset.top_k(3):
        print(" ", g)
        print(f"    quaternion: {g.quaternion()}")

    print("\n=== Box grasps ===")
    gset = selector.grasp_candidates(box, object_id=1)
    for g in gset.top_k(3):
        print(" ", g)

    print("\n=== Clearing sequence (nearest first) ===")
    seq = selector.plan_clearing_sequence([mug, box], table_height=0.0)
    for i, g in enumerate(seq):
        print(f"  [{i+1}] {g}")
