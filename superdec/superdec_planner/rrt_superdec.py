from __future__ import annotations
import time
from functools import partial
from typing import Dict, List, Literal, Optional
import numpy as np
from ompl import base as ob
from ompl import geometric as og


PlannerName = Literal["rrtstar", "rrtconnect", "rrt"]

class PathPlanner:
    """
    OMPL-based 3D path planner with distance-field collision checks.

    Expects `supdec_scene` to provide:
      - get_distances_and_closest_points(points: (N,3) array)
        -> (distances: (N,), closest_points: (N,3))
        Distances must be positive outside obstacles, negative/zero inside.
    """

    def __init__(self, collision_radius: float = 0.15, validity_resolution: float = 0.01):
        # 3D Euclidean state space
        self.sp = ob.RealVectorStateSpace(3)

        # Initialize with a sane default bound (updated later via update_sp)
        bounds = ob.RealVectorBounds(3)
        for i in range(3):
            bounds.setLow(i, -5.0)
            bounds.setHigh(i, 5.0)
        self.sp.setBounds(bounds)

        # OMPL setup
        self.ss = og.SimpleSetup(self.sp)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(PathPlanner.is_state_valid, self)))
        self.sp.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(validity_resolution)

        # External scene 
        self.supdec_scene = None  # set via update_sp

        # Planner params/state
        self.collision_radius: float = float(collision_radius)

        # Start/goal storage
        self.start_pos: Optional[np.ndarray] = None
        self.goal_pos: Optional[np.ndarray] = None
        self.start_quat: Optional[np.ndarray] = None
        self.goal_quat: Optional[np.ndarray] = None

        # Timing for validity checks
        self.valid_count: int = 0
        self.cumulative_time: float = 0.0

        # Cache last solution
        self._last_path = None

    # Configuration & utilities
    def reset_timing(self):
        """Reset validity timing counters."""
        self.valid_count = 0
        self.cumulative_time = 0.0

    def update_collision_radius(self, collision_radius: float):
        """Set the robot's collision radius (meters)."""
        if collision_radius < 0:
            raise ValueError("collision_radius must be >= 0")
        self.collision_radius = float(collision_radius)

    def is_state_valid(self, state: ob.State):
        """
        Validity via distance field:
        A state is valid iff distance(to nearest obstacle) > collision_radius.
        """
        if self.supdec_scene is None:
            # If scene not provided yet, treat everything as valid to avoid hard failure.
            return True

        state_3dpt = np.array([[float(state[0]), float(state[1]), float(state[2])]], dtype=np.float32)

        start_time = time.time()
        dist, _ = self.supdec_scene.get_distances_and_closest_points(state_3dpt)
        self.valid_count += 1
        self.cumulative_time += (time.time() - start_time)

        return float(dist[0]) > self.collision_radius

    def _nearest_valid_point(
        self,
        p: np.ndarray,
        eps: float = 1e-3,
        max_iters: int = 5,
        clamp_to_bounds: bool = True,
    ):
        """
        If p is too close to obstacles, push it outward along the vector away from the closest obstacle point
        until distance > collision_radius + eps (or bounds stop us).
        Returns (p_fixed, final_distance_to_obstacle).
        """
        assert p.shape == (3,), "Input point must be shape (3,)"

        if self.supdec_scene is None:
            return p.copy(), np.inf  # no scene -> assume valid

        target_clearance = self.collision_radius + eps
        q = p.astype(np.float32).copy()

        for _ in range(max_iters):
            dist, cp = self.supdec_scene.get_distances_and_closest_points(q[None, :])
            d = float(dist[0])
            if d > target_clearance:
                return q, d

            cpt = cp[0].astype(np.float32)
            v = q - cpt
            norm = float(np.linalg.norm(v))
            if norm < 1e-9:
                v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                norm = 1.0

            dir_out = v / norm
            step = (target_clearance - d) + 0.5 * eps  
            q = q + step * dir_out

            if clamp_to_bounds:
                # Keep within configured bounds
                b = self.sp.getBounds()
                q[0] = min(max(q[0], b.low[0]), b.high[0])
                q[1] = min(max(q[1], b.low[1]), b.high[1])
                q[2] = min(max(q[2], b.low[2]), b.high[2])

        dist, _ = self.supdec_scene.get_distances_and_closest_points(q[None, :])
        return q, float(dist[0])

    # Scene & bounds
    def update_sp(self, bound: Dict[str, float], supdec_scene):
        """
        Update space bounds from a voxelgrid bound and attach the scene.
        `bound` requires keys: low_x, high_x, low_y, high_y, low_z, high_z
        """
        required = {"low_x", "high_x", "low_y", "high_y", "low_z", "high_z"}
        if not required.issubset(bound):
            missing = sorted(required - set(bound))
            raise KeyError(f"Missing bound keys: {missing}")

        # Optional small epsilon in case you want to pad the domain a bit.
        epsilon = 0.0

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, float(bound["low_x"]) - epsilon)
        bounds.setHigh(0, float(bound["high_x"]) + epsilon)
        bounds.setLow(1, float(bound["low_y"]) - epsilon)
        bounds.setHigh(1, float(bound["high_y"]) + epsilon)
        bounds.setLow(2, float(bound["low_z"]) - epsilon)
        bounds.setHigh(2, float(bound["high_z"]) + epsilon)

        self.sp.setBounds(bounds)
        self.supdec_scene = supdec_scene
        # Re-setup is cheap; ensures internal consistency if bounds changed
        self.sp.setup()
        self.ss.getSpaceInformation().setup()
        print("Updated the scene and bounds.")

    def get_bounds(self):
        b = self.sp.getBounds()
        return {
            "low_x": b.low[0],
            "high_x": b.high[0],
            "low_y": b.low[1],
            "high_y": b.high[1],
            "low_z": b.low[2],
            "high_z": b.high[2],
        }

    # Start/Goal
    def update_start_goal(
        self,
        start: Dict[str, np.ndarray],
        goal: Dict[str, np.ndarray],
        snap_to_valid: bool = True,
        snap_eps: float = 1e-3,
    ):
        """
        Update start/goal. If `snap_to_valid`, nudge them to the nearest valid points if too close to obstacles.
        Expects dicts: {"pos": (3,), "quat": (4,)}; stored as numpy arrays.
        """
        self.start_pos = np.asarray(start["pos"], dtype=np.float32).copy()
        self.goal_pos = np.asarray(goal["pos"], dtype=np.float32).copy()
        self.start_quat = np.asarray(start["quat"], dtype=np.float32).copy()
        self.goal_quat = np.asarray(goal["quat"], dtype=np.float32).copy()

        if snap_to_valid:
            self.start_pos, d_s = self._nearest_valid_point(self.start_pos, eps=snap_eps)
            self.goal_pos, d_g = self._nearest_valid_point(self.goal_pos, eps=snap_eps)
            # Optional: warn if still not valid (e.g., blocked by bounds)
            if d_s <= self.collision_radius:
                print(f"[warn] Start still in collision after snap (dist={d_s:.4f}).")
            if d_g <= self.collision_radius:
                print(f"[warn] Goal still in collision after snap (dist={d_g:.4f}).")

        # Build OMPL states
        start_state = ob.State(self.sp)
        start_state()[0], start_state()[1], start_state()[2] = map(float, self.start_pos)

        goal_state = ob.State(self.sp)
        goal_state()[0], goal_state()[1], goal_state()[2] = map(float, self.goal_pos)

        # A small goal region radius; tune as needed
        self.ss.setStartAndGoalStates(start_state, goal_state, 0.1)
        print("Updated start and goal.",
              "start:", self.start_pos.tolist(),
              "goal:", self.goal_pos.tolist())
        return True

    def get_start_goal(self) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            "start": {"pos": self.start_pos, "quat": self.start_quat},
            "goal": {"pos": self.goal_pos, "quat": self.goal_quat},
        }

    # Planning
    def _make_planner(self, name: PlannerName):
        si = self.ss.getSpaceInformation()
        if name == "rrtstar":
            planner = og.RRTstar(si)
            return planner
        if name == "rrtconnect":
            planner = og.RRTConnect(si)
            return planner
        if name == "rrt":
            planner = og.RRT(si)
            return planner
        raise ValueError(f"Unknown planner '{name}'")

    def get_motion_check_resolution(self):
        return self.ss.getSpaceInformation().getStateValidityCheckingResolution()

    def set_motion_check_resolution(self, resolution: float):
        if resolution <= 0:
            raise ValueError("resolution must be > 0")
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(resolution)

    def solve(self, time_limit: float = 5.0, method: PlannerName = "rrtstar"):
        """
        Run the planner. Returns True if a solution is found within `time_limit` seconds.
        """
        if self.start_pos is None or self.goal_pos is None:
            raise RuntimeError("Start/goal not set. Call update_start_goal first.")
        if self.supdec_scene is None:
            print("[warn] supdec_scene is None; validity checks may be meaningless.")

        planner = self._make_planner(method)
        self.ss.setPlanner(planner)

        solved = self.ss.solve(float(time_limit))
        if solved:
            # Path shortening can significantly improve quality
            self.ss.simplifySolution()
            self._last_path = self.ss.getSolutionPath()
        else:
            self._last_path = None
            print("Solver failed to find a solution.")
        return bool(solved)

    def get_avg_valid_time(self) -> float:
        if self.valid_count == 0:
            return 0.0
        return self.cumulative_time / self.valid_count

    # Solution extraction
    def get_solution(self):
        """
        Returns a list of waypoints with pos/quat.
        - First pose uses start quaternion.
        - Intermediate poses use [0,0,0,1].
        - Final pose uses goal quaternion.
        """
        try:
            path: og.PathGeometric = self.ss.getSolutionPath()
        except Exception:
            print("No solution found.")
            return None

        if path is None:
            print("No solution found.")
            return None

        states = path.getStates()
        if not states:
            print("Solution path is empty.")
            return None

        solution: List[Dict[str, List[float]]] = []
        for i, st in enumerate(states):
            if i == 0:
                pos = [float(self.start_pos[0]), float(self.start_pos[1]), float(self.start_pos[2])]
                quat = [float(self.start_quat[0]), float(self.start_quat[1]),
                        float(self.start_quat[2]), float(self.start_quat[3])]
            else:
                pos = [float(st[0]), float(st[1]), float(st[2])]
                # default neutral quaternion
                quat = [0.0, 0.0, 0.0, 1.0]
                if i == len(states) - 1:
                    quat = [float(self.goal_quat[0]), float(self.goal_quat[1]),
                            float(self.goal_quat[2]), float(self.goal_quat[3])]
            solution.append({"pos": pos, "quat": quat})
        return solution

    def get_solution_positions(self):
        """
        Returns Nx3 numpy array of positions along the solution, or None.
        """
        try:
            path: og.PathGeometric = self.ss.getSolutionPath()
        except Exception:
            return None
        if path is None:
            return None
        states = path.getStates()
        if not states:
            return None
        out = np.array([[float(s[0]), float(s[1]), float(s[2])] for s in states], dtype=np.float32)
        return out
