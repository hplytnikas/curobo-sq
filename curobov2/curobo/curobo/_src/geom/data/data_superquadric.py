# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""GPU data storage and SDF functions for superquadric obstacles.

This module provides:
- SuperquadricData: Python dataclass owning GPU tensors
- SuperquadricDataWarp: Warp struct for kernel access
- is_obs_enabled, load_obstacle_transform, compute_local_sdf,
  compute_local_sdf_with_grad: Generic kernel overloads

SDF math (Newton radial projection):
  F(p) = ((|x/sx|^{2/ε₂} + |y/sy|^{2/ε₂})^{ε₂/ε₁} + |z/sz|^{2/ε₁})
  F = 1 on surface, >1 outside, <1 inside.

  Outside (F ≥ 1): Taubin approximation — sdf = (F−1)/‖∇F‖
  Inside  (F < 1): Newton radial projection — find λ s.t. F(λ·p)=1,
                   then sdf = (1−λ)·|p|

Sign convention: positive outside, negative inside (matches CuRobo standard).
The sphere radius is NOT subtracted here; the generic kernel handles that via
penetration = -sdf + sphere_radius_adjusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import warp as wp

from curobo._src.geom.data.helper_pose import (
    get_obs_idx,
    load_transform_from_inv_pose,
)
from curobo._src.geom.types import SceneCfg, Superquadric
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


# =============================================================================
# Warp Struct
# =============================================================================


@wp.struct
class SuperquadricDataWarp:
    """Warp struct for superquadric obstacle data.

    Attributes:
        params: Semi-axes and shape exponents [sx, sy, sz, eps1, eps2, pad, pad, pad].
            Shape: (num_envs * max_n, 8).
        inv_pose: Inverse pose [x, y, z, qw, qx, qy, qz, pad]. Shape: (num_envs * max_n, 8).
        enable: Enable flag per SQ. 1 = active, 0 = disabled.
        n_per_env: Number of active SQs per environment.
        max_n: Maximum SQs per environment.
        num_envs: Number of environments.
    """

    params: wp.array2d(dtype=wp.float32)
    inv_pose: wp.array2d(dtype=wp.float32)
    enable: wp.array(dtype=wp.uint8)
    n_per_env: wp.array(dtype=wp.int32)
    max_n: wp.int32
    num_envs: wp.int32


# =============================================================================
# Python Dataclass
# =============================================================================


@dataclass
class SuperquadricData:
    """GPU tensor storage for superquadric obstacles.

    Tensor layouts:
        - params: (num_envs, max_n, 8) - [sx, sy, sz, eps1, eps2, pad, pad, pad]
        - inv_pose: (num_envs, max_n, 8) - inverse pose [x, y, z, qw, qx, qy, qz, pad]
        - enable: (num_envs, max_n) - uint8 flags
        - count: (num_envs,) - int32 active counts
    """

    params: torch.Tensor
    inv_pose: torch.Tensor
    enable: torch.Tensor
    count: torch.Tensor
    names: List[List[Optional[str]]]
    max_n: int
    num_envs: int
    device_cfg: DeviceCfg

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def create_cache(
        cls,
        max_n: int,
        num_envs: int,
        device_cfg: DeviceCfg,
    ) -> "SuperquadricData":
        """Create an empty cache for superquadric obstacles."""
        params = torch.zeros(
            (num_envs, max_n, 8),
            dtype=device_cfg.dtype,
            device=device_cfg.device,
        )
        # Default: unit sphere (sx=sy=sz=0.1, eps1=eps2=1.0)
        params[..., 0] = 0.1
        params[..., 1] = 0.1
        params[..., 2] = 0.1
        params[..., 3] = 1.0
        params[..., 4] = 1.0

        inv_pose = torch.zeros(
            (num_envs, max_n, 8),
            dtype=device_cfg.dtype,
            device=device_cfg.device,
        )
        inv_pose[..., 3] = 1.0  # Identity quaternion (qw=1)

        enable = torch.zeros((num_envs, max_n), dtype=torch.uint8, device=device_cfg.device)
        count = torch.zeros((num_envs,), device=device_cfg.device, dtype=torch.int32)
        names = [[None for _ in range(max_n)] for _ in range(num_envs)]

        return cls(
            params=params,
            inv_pose=inv_pose,
            enable=enable,
            count=count,
            names=names,
            max_n=max_n,
            num_envs=num_envs,
            device_cfg=device_cfg,
        )

    @classmethod
    def from_scene_cfg(
        cls,
        scene_cfg: SceneCfg,
        device_cfg: DeviceCfg,
        env_idx: int = 0,
        num_envs: int = 1,
        max_n: Optional[int] = None,
    ) -> "SuperquadricData":
        """Create SuperquadricData from a scene configuration."""
        sqs = scene_cfg.superquadric if hasattr(scene_cfg, "superquadric") else []
        num_sqs = len(sqs) if sqs else 0
        if max_n is None:
            max_n = max(num_sqs, 1)

        instance = cls.create_cache(max_n, num_envs, device_cfg)
        if num_sqs > 0:
            instance.load_batch(sqs, env_idx)
        return instance

    @classmethod
    def from_batch_scene_cfg(
        cls,
        scene_cfg_list: List[SceneCfg],
        device_cfg: DeviceCfg,
        max_n: Optional[int] = None,
    ) -> "SuperquadricData":
        """Create SuperquadricData from a list of scene configurations."""
        num_envs = len(scene_cfg_list)
        sq_counts = [
            len(cfg.superquadric) if (hasattr(cfg, "superquadric") and cfg.superquadric) else 0
            for cfg in scene_cfg_list
        ]
        if max_n is None:
            max_n = max(sq_counts) if sq_counts else 1

        instance = cls.create_cache(max_n, num_envs, device_cfg)
        for env_idx, scene_cfg in enumerate(scene_cfg_list):
            sqs = scene_cfg.superquadric if (hasattr(scene_cfg, "superquadric") and scene_cfg.superquadric) else []
            if sqs:
                instance.load_batch(sqs, env_idx)
        return instance

    # -------------------------------------------------------------------------
    # Load / Add Methods
    # -------------------------------------------------------------------------

    def load_batch(self, sqs: List[Superquadric], env_idx: int) -> None:
        """Load a batch of superquadrics into an environment, replacing existing ones."""
        num_sqs = len(sqs)
        if num_sqs > self.max_n:
            log_and_raise(f"Cannot load {num_sqs} superquadrics, max cache size is {self.max_n}")
        if num_sqs == 0:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            return

        for i, sq in enumerate(sqs):
            w_T_sq = Pose.from_list(sq.pose, device_cfg=self.device_cfg)
            sq_T_w = w_T_sq.inverse()
            inv_pose_vec = sq_T_w.get_pose_vector()

            self.params[env_idx, i, 0] = float(sq.radii[0])
            self.params[env_idx, i, 1] = float(sq.radii[1])
            self.params[env_idx, i, 2] = float(sq.radii[2])
            self.params[env_idx, i, 3] = float(sq.shape[0])
            self.params[env_idx, i, 4] = float(sq.shape[1])
            self.inv_pose[env_idx, i, :7] = inv_pose_vec
            self.enable[env_idx, i] = 1
            self.names[env_idx][i] = sq.name

        self.enable[env_idx, num_sqs:] = 0
        self.count[env_idx] = num_sqs

    def add(self, sq: Superquadric, env_idx: int = 0) -> int:
        """Add a single superquadric to an environment."""
        current_count = int(self.count[env_idx].item())
        if current_count >= self.max_n:
            log_and_raise(f"Cannot add superquadric, cache is full ({self.max_n})")
        if sq.name in self.names[env_idx]:
            log_and_raise(f"Superquadric already exists with name: {sq.name}")

        w_T_sq = Pose.from_list(sq.pose, device_cfg=self.device_cfg)
        sq_T_w = w_T_sq.inverse()
        inv_pose_vec = sq_T_w.get_pose_vector()

        self.params[env_idx, current_count, 0] = float(sq.radii[0])
        self.params[env_idx, current_count, 1] = float(sq.radii[1])
        self.params[env_idx, current_count, 2] = float(sq.radii[2])
        self.params[env_idx, current_count, 3] = float(sq.shape[0])
        self.params[env_idx, current_count, 4] = float(sq.shape[1])
        self.inv_pose[env_idx, current_count, :7] = inv_pose_vec
        self.enable[env_idx, current_count] = 1
        self.names[env_idx][current_count] = sq.name
        self.count[env_idx] += 1

        return current_count

    # -------------------------------------------------------------------------
    # Update Methods
    # -------------------------------------------------------------------------

    def update_pose(
        self,
        name: str,
        w_obj_pose: Optional[Pose] = None,
        obj_w_pose: Optional[Pose] = None,
        env_idx: int = 0,
    ) -> None:
        """Update the pose of an existing superquadric."""
        if w_obj_pose is None and obj_w_pose is None:
            log_and_raise("Either w_obj_pose or obj_w_pose must be provided")
        idx = self.get_idx(name, env_idx)
        if w_obj_pose is not None:
            obj_w_pose = w_obj_pose.inverse()
        self.inv_pose[env_idx, idx, :7] = obj_w_pose.get_pose_vector()

    def set_enabled(self, name: str, enabled: bool, env_idx: int = 0) -> None:
        """Enable or disable a superquadric."""
        idx = self.get_idx(name, env_idx)
        self.enable[env_idx, idx] = int(enabled)

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_idx(self, name: str, env_idx: int = 0) -> int:
        """Get the index of a superquadric by name."""
        return self.names[env_idx].index(name)

    def get_active_count(self, env_idx: int = 0) -> int:
        """Get the number of active superquadrics."""
        return int(self.count[env_idx].item())

    def get_names(self, env_idx: int = 0) -> List[str]:
        """Get the names of all active superquadrics."""
        count = int(self.count[env_idx].item())
        return self.names[env_idx][:count]

    def clear(self, env_idx: Optional[int] = None) -> None:
        """Clear all superquadrics from one or all environments."""
        if env_idx is not None:
            self.enable[env_idx, :] = 0
            self.count[env_idx] = 0
            self.names[env_idx] = [None] * self.max_n
        else:
            self.enable[:] = 0
            self.count[:] = 0
            self.names = [[None] * self.max_n for _ in range(self.num_envs)]

    # -------------------------------------------------------------------------
    # Warp Conversion
    # -------------------------------------------------------------------------

    def to_warp(self) -> SuperquadricDataWarp:
        """Convert to Warp struct for kernel launches."""
        s = SuperquadricDataWarp()
        s.params = wp.from_torch(self.params.view(-1, 8), dtype=wp.float32)
        s.inv_pose = wp.from_torch(self.inv_pose.view(-1, 8), dtype=wp.float32)
        s.enable = wp.from_torch(self.enable.view(-1), dtype=wp.uint8)
        s.n_per_env = wp.from_torch(self.count.view(-1), dtype=wp.int32)
        s.max_n = self.max_n
        s.num_envs = self.num_envs
        return s


# =============================================================================
# SDF Implementation (Generic Kernel Overloads)
# =============================================================================

_SDF_EPS = wp.float32(1e-9)
_SDF_EPS2 = wp.float32(1e-8)


@wp.func
def is_obs_enabled(
    obs_set: SuperquadricDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.bool:
    if local_idx >= obs_set.n_per_env[env_idx]:
        return False
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return obs_set.enable[flat_idx] == wp.uint8(1)


@wp.func
def load_obstacle_transform(
    obs_set: SuperquadricDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
) -> wp.transform:
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    return load_transform_from_inv_pose(obs_set.inv_pose, flat_idx)


@wp.func
def compute_local_sdf_with_grad(
    obs_set: SuperquadricDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.vec4:
    """Compute superquadric SDF and local-frame gradient.

    The query point must already be in the SQ's local frame.
    Returns vec4(sdf, grad_x, grad_y, grad_z) where the gradient is in local frame
    and points outward from the obstacle surface.

    SDF sign: positive outside, negative inside.
    """
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)

    sx = obs_set.params[flat_idx, 0]
    sy = obs_set.params[flat_idx, 1]
    sz = obs_set.params[flat_idx, 2]
    eps1 = obs_set.params[flat_idx, 3]
    eps2 = obs_set.params[flat_idx, 4]

    lx = local_pt[0]
    ly = local_pt[1]
    lz = local_pt[2]

    inv_e1 = wp.float32(1.0) / eps1
    p1 = wp.float32(2.0) * inv_e1
    p2 = wp.float32(2.0) / eps2
    e_ratio = eps2 * inv_e1

    # Compute F at query point
    ax = wp.max(wp.abs(lx) / sx, _SDF_EPS)
    ay = wp.max(wp.abs(ly) / sy, _SDF_EPS)
    az = wp.max(wp.abs(lz) / sz, _SDF_EPS)

    lax = wp.log(ax)
    lay = wp.log(ay)
    laz = wp.log(az)

    xt = wp.exp(p2 * lax)
    yt = wp.exp(p2 * lay)
    zt = wp.exp(p1 * laz)
    sum_xy = xt + yt
    lsum = wp.log(wp.max(sum_xy, _SDF_EPS))
    F = wp.exp(e_ratio * lsum) + zt

    # Outputs initialized to zero (updated in branches below)
    sdf = wp.float32(0.0)
    gx = wp.float32(0.0)
    gy = wp.float32(0.0)
    gz = wp.float32(0.0)

    if F < wp.float32(1.0):
        # ── Inside: Newton radial projection ─────────────────────────────────
        p_len = wp.sqrt(lx * lx + ly * ly + lz * lz)

        if p_len < wp.float32(1e-6):
            # Near-center fallback: return negative of smallest semi-axis
            min_semi = wp.min(sx, wp.min(sy, sz))
            sdf = -min_semi
            gz = wp.float32(1.0)  # arbitrary outward direction
        else:
            # Seed λ using F(λp) ≈ λ^(2/ε₁)·F(p) → λ_star ≈ F^(-ε₁/2)
            F_safe = wp.max(F, wp.float32(1e-30))
            lambda_ = wp.max(
                wp.float32(1.0),
                wp.pow(F_safe, wp.float32(-0.5) * eps1),
            )

            # Newton iterations: λ ← λ - (F(λp)-1) / (∇F(λp)·p)
            for _i in range(4):
                qx = lambda_ * lx
                qy = lambda_ * ly
                qz = lambda_ * lz

                nax = wp.max(wp.abs(qx) / sx, _SDF_EPS)
                nay = wp.max(wp.abs(qy) / sy, _SDF_EPS)
                naz = wp.max(wp.abs(qz) / sz, _SDF_EPS)

                nlax = wp.log(nax)
                nlay = wp.log(nay)
                nlaz = wp.log(naz)

                nxt = wp.exp(p2 * nlax)
                nyt = wp.exp(p2 * nlay)
                nsum = nxt + nyt
                nlsum = wp.log(wp.max(nsum, _SDF_EPS))
                nF = wp.exp(e_ratio * nlsum) + wp.exp(p1 * nlaz)

                # Safe sign (never zero)
                nsx_sign = wp.float32(1.0)
                if qx < wp.float32(0.0):
                    nsx_sign = wp.float32(-1.0)
                nsy_sign = wp.float32(1.0)
                if qy < wp.float32(0.0):
                    nsy_sign = wp.float32(-1.0)
                nsz_sign = wp.float32(1.0)
                if qz < wp.float32(0.0):
                    nsz_sign = wp.float32(-1.0)

                ps = wp.exp((e_ratio - wp.float32(1.0)) * nlsum)
                c = wp.float32(2.0) * inv_e1

                ngx = c / sx * nsx_sign * ps * wp.exp((p2 - wp.float32(1.0)) * nlax)
                ngy = c / sy * nsy_sign * ps * wp.exp((p2 - wp.float32(1.0)) * nlay)
                ngz = c / sz * nsz_sign * wp.exp((p1 - wp.float32(1.0)) * nlaz)

                df = ngx * lx + ngy * ly + ngz * lz
                delta = -(nF - wp.float32(1.0)) / (df + _SDF_EPS2)
                lambda_ = lambda_ + delta

            sdf = (wp.float32(1.0) - lambda_) * p_len

            # Final gradient at surface point λ·p_local
            sqx = lambda_ * lx
            sqy = lambda_ * ly
            sqz = lambda_ * lz

            snax = wp.max(wp.abs(sqx) / sx, _SDF_EPS)
            snay = wp.max(wp.abs(sqy) / sy, _SDF_EPS)
            snaz = wp.max(wp.abs(sqz) / sz, _SDF_EPS)

            snlax = wp.log(snax)
            snlay = wp.log(snay)
            snlaz = wp.log(snaz)

            snsum = wp.exp(p2 * snlax) + wp.exp(p2 * snlay)
            snlsum = wp.log(wp.max(snsum, _SDF_EPS))
            sps = wp.exp((e_ratio - wp.float32(1.0)) * snlsum)
            sc = wp.float32(2.0) * inv_e1

            ssx_sign = wp.float32(1.0)
            if sqx < wp.float32(0.0):
                ssx_sign = wp.float32(-1.0)
            ssy_sign = wp.float32(1.0)
            if sqy < wp.float32(0.0):
                ssy_sign = wp.float32(-1.0)
            ssz_sign = wp.float32(1.0)
            if sqz < wp.float32(0.0):
                ssz_sign = wp.float32(-1.0)

            gx = sc / sx * ssx_sign * sps * wp.exp((p2 - wp.float32(1.0)) * snlax)
            gy = sc / sy * ssy_sign * sps * wp.exp((p2 - wp.float32(1.0)) * snlay)
            gz = sc / sz * ssz_sign * wp.exp((p1 - wp.float32(1.0)) * snlaz)

            inv_g = wp.float32(1.0) / wp.sqrt(gx * gx + gy * gy + gz * gz + _SDF_EPS2)
            gx = gx * inv_g
            gy = gy * inv_g
            gz = gz * inv_g

    else:
        # ── Outside: Taubin approximation + conservative lower bounds ─────────
        ox_sign = wp.float32(1.0)
        if lx < wp.float32(0.0):
            ox_sign = wp.float32(-1.0)
        oy_sign = wp.float32(1.0)
        if ly < wp.float32(0.0):
            oy_sign = wp.float32(-1.0)
        oz_sign = wp.float32(1.0)
        if lz < wp.float32(0.0):
            oz_sign = wp.float32(-1.0)

        ops = wp.exp((e_ratio - wp.float32(1.0)) * lsum)
        oc = wp.float32(2.0) * inv_e1

        gx = oc / sx * ox_sign * ops * wp.exp((p2 - wp.float32(1.0)) * lax)
        gy = oc / sy * oy_sign * ops * wp.exp((p2 - wp.float32(1.0)) * lay)
        gz = oc / sz * oz_sign * wp.exp((p1 - wp.float32(1.0)) * laz)
        g2 = gx * gx + gy * gy + gz * gz

        inv_g = wp.float32(1.0) / wp.sqrt(g2 + _SDF_EPS2)
        sdf = (F - wp.float32(1.0)) * inv_g

        # Normalize gradient
        gx = gx * inv_g
        gy = gy * inv_g
        gz = gz * inv_g

        # Lower bound 1: distance from origin to point minus largest semi-axis
        # (rotation-invariant, since |local_pt| = |world_pt - center|)
        r_outer = wp.max(sx, wp.max(sy, sz))
        p_len_o = wp.sqrt(lx * lx + ly * ly + lz * lz)
        lb = p_len_o - r_outer

        # Lower bound 2: distance from point to AABB in local SQ frame
        dx_box = wp.max(wp.abs(lx) - sx, wp.float32(0.0))
        dy_box = wp.max(wp.abs(ly) - sy, wp.float32(0.0))
        dz_box = wp.max(wp.abs(lz) - sz, wp.float32(0.0))
        lb_box = wp.sqrt(dx_box * dx_box + dy_box * dy_box + dz_box * dz_box)

        sdf = wp.max(sdf, wp.max(lb, lb_box))

    return wp.vec4(sdf, gx, gy, gz)


@wp.func
def compute_local_sdf(
    obs_set: SuperquadricDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.float32:
    result = compute_local_sdf_with_grad(obs_set, env_idx, local_idx, local_pt)
    return result[0]