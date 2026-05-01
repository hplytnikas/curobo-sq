/*
 * /*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

/* ═══════════════════════════════════════════════════════════════════════════
 * SQData struct: 48 bytes per SQ descriptor, optimised for coalesced access and
* shared memory storage in the CUDA kernels.
* ═══════════════════════════════════════════════════════════════════════════ */

struct __align__(16) SQData {
    float cx, cy, cz;      // centre
    float sx, sy, sz;      // semi-axes
    float eps1, eps2;      // shape exponents
    float qw, qx, qy, qz; // orientation quaternion (local←world)
    float hx, hy, hz;     // world-frame AABB half-extents (precomputed in pack_env_sq)
    float _pad;            // padding to 64 bytes
};
static_assert(sizeof(SQData) == 64, "SQData must be 64 bytes");

/* ═══════════════════════════ Device helpers ════════════════════════════════ */

// Safe log: prevents log(0) → -inf → NaN propagation.
__device__ __forceinline__ float flog_safe(float x)
{
    return __logf(fmaxf(x, 1e-9f));
}

// Rotate vector v by conjugate of quaternion q (i.e. world→local transform)
__device__ __forceinline__
void rotate_by_quat_inv(
    float vx, float vy, float vz,
    float qw, float qx, float qy, float qz,
    float& ox, float& oy, float& oz)
{
    // q* = (qw, -qx, -qy, -qz)
    // v' = q* ⊗ v ⊗ q
    // Expanded sandwich product:
    float tx = 2.f * ((-qy) * vz - (-qz) * vy);
    float ty = 2.f * ((-qz) * vx - (-qx) * vz);
    float tz = 2.f * ((-qx) * vy - (-qy) * vx);
    ox = vx + qw * tx + ((-qy) * tz - (-qz) * ty);
    oy = vy + qw * ty + ((-qz) * tx - (-qx) * tz);
    oz = vz + qw * tz + ((-qx) * ty - (-qy) * tx);
}

// Rotate vector v by quaternion q (i.e. local→world transform)
// v' = q ⊗ v ⊗ q*
__device__ __forceinline__
void rotate_by_quat_fwd(
    float vx, float vy, float vz,
    float qw, float qx, float qy, float qz,
    float& ox, float& oy, float& oz)
{
    float tx = 2.f * (qy * vz - qz * vy);
    float ty = 2.f * (qz * vx - qx * vz);
    float tz = 2.f * (qx * vy - qy * vx);
    ox = vx + qw * tx + (qy * tz - qz * ty);
    oy = vy + qw * ty + (qz * tx - qx * tz);
    oz = vz + qw * tz + (qx * ty - qy * tx);
}


/* sq_aabb_miss: broad-phase rejection using the precomputed world-frame AABB.
 *
 * hx/hy/hz are stored in SQData (precomputed by pack_env_sq from the rotation
 * matrix absolute values: h_i = Σ_j |R_ij| · s_j).  Conservative for all eps.
 *
 * Returns true when a sphere (center px,py,pz; reach = sphere_radius + margin)
 * is provably outside the AABB and the sq_sdf call can be skipped safely.
 * Only used in cost-accumulation kernels where zero-cost pairs are skippable;
 * NOT used in min-distance kernels (they need the full ESDF field). */
__device__ __forceinline__
bool sq_aabb_miss(const float px, const float py, const float pz,
                  const float reach, const SQData& sq)
{
    return fmaxf(fabsf(px - sq.cx) - sq.hx,
                 fmaxf(fabsf(py - sq.cy) - sq.hy,
                       fabsf(pz - sq.cz) - sq.hz)) > reach;
}

/* ═══════════════════════════ Newton radial solve ════════════════════════════
 *
 * For points INSIDE the SQ (F < 1), the Taubin approximation and lb_box both
 * return a constant value of −r_sphere, giving a flat cost landscape that
 * defeats gradient-based trajectory optimisation.
 *
 * Instead we use Newton iteration to find λ such that F(λ·p_local) = 1.
 * The radial signed distance is then (1 − λ)·|p_local|, which is:
 *   negative (inside)  when λ > 1
 *   positive (outside) when λ < 1
 *
 * Iteration: λ ← λ − (F(λp) − 1) / (∇F(λp) · p),  starting from λ = 1.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* sq_newton_lambda: returns λ for the radial distance (no gradient output).
 *
 * F_init: F evaluated at the original query point (passed from sq_sdf to
 * avoid recomputation).  Used to set a near-optimal starting λ via the
 * scaling identity F(λ·p) ≈ λ^(2/ε₁)·F(p), giving λ_init = F^(−ε₁/2).
 * This prevents the catastrophic first-step overshoot that occurs for boxy
 * shapes (small ε) when starting from λ=1, where F is nearly zero and
 * ∇F·p is exponentially small, causing the Newton step to diverge to ~10⁴
 * and triggering float32 exp() overflow → NaN → false "no-collision". */
__device__ __forceinline__
float sq_newton_lambda(
    const float lx, const float ly, const float lz,
    const SQData& sq,
    const float F_init)
{
    const float inv_e1  = __frcp_rn(sq.eps1);
    const float p1      = 2.f * inv_e1;
    const float p2      = 2.f * __frcp_rn(sq.eps2);
    const float e_ratio = sq.eps2 * inv_e1;

    // Better initial λ: F(λp) ≈ λ^(2/ε₁)·F(p)  →  λ_star ≈ F^(-ε₁/2)
    // For λ inside (F < 1) this gives λ > 1, landing near the surface.
    float lambda = fmaxf(1.f, __powf(fmaxf(F_init, 1e-30f), -0.5f * sq.eps1));

    for (int i = 0; i < 4; ++i) {
        const float qx = lambda * lx;
        const float qy = lambda * ly;
        const float qz = lambda * lz;

        const float ax = fmaxf(fabsf(qx) * __frcp_rn(sq.sx), 1e-9f);
        const float ay = fmaxf(fabsf(qy) * __frcp_rn(sq.sy), 1e-9f);
        const float az = fmaxf(fabsf(qz) * __frcp_rn(sq.sz), 1e-9f);

        const float lax  = flog_safe(ax);
        const float lay  = flog_safe(ay);
        const float laz  = flog_safe(az);
        const float xt   = __expf(p2 * lax);
        const float yt   = __expf(p2 * lay);
        const float zt   = __expf(p1 * laz);
        const float sum  = xt + yt;
        const float lsum = flog_safe(sum);
        const float F    = __expf(e_ratio * lsum) + zt;

        const float sx_sign = copysignf(1.f, qx);
        const float sy_sign = copysignf(1.f, qy);
        const float sz_sign = copysignf(1.f, qz);
        const float ps      = __expf((e_ratio - 1.f) * lsum);
        const float c       = 2.f * inv_e1;

        const float gx = c * __frcp_rn(sq.sx) * sx_sign * ps * __expf((p2 - 1.f) * lax);
        const float gy = c * __frcp_rn(sq.sy) * sy_sign * ps * __expf((p2 - 1.f) * lay);
        const float gz = c * __frcp_rn(sq.sz) * sz_sign       * __expf((p1 - 1.f) * laz);

        const float df    = fmaf(gx, lx, fmaf(gy, ly, gz * lz));
        const float delta = -(F - 1.f) / (df + 1e-8f);
        lambda += delta;
        if (fabsf(delta) < 1e-4f) break;
    }

    return lambda;
}

/* sq_newton_lambda_and_surf_grad: returns λ AND ∇F at the final surface point.
 * Used by sq_sdf_and_normal so the gradient path also gets a correct normal.
 * F_init: same role as in sq_newton_lambda — seeds a better starting λ. */
__device__ __forceinline__
float sq_newton_lambda_and_surf_grad(
    const float lx, const float ly, const float lz,
    const SQData& sq,
    float& surf_gx, float& surf_gy, float& surf_gz,
    const float F_init)
{
    const float inv_e1  = __frcp_rn(sq.eps1);
    const float p1      = 2.f * inv_e1;
    const float p2      = 2.f * __frcp_rn(sq.eps2);
    const float e_ratio = sq.eps2 * inv_e1;

    float lambda = fmaxf(1.f, __powf(fmaxf(F_init, 1e-30f), -0.5f * sq.eps1));

    for (int i = 0; i < 4; ++i) {
        const float qx = lambda * lx;
        const float qy = lambda * ly;
        const float qz = lambda * lz;

        const float ax = fmaxf(fabsf(qx) * __frcp_rn(sq.sx), 1e-9f);
        const float ay = fmaxf(fabsf(qy) * __frcp_rn(sq.sy), 1e-9f);
        const float az = fmaxf(fabsf(qz) * __frcp_rn(sq.sz), 1e-9f);

        const float lax  = flog_safe(ax);
        const float lay  = flog_safe(ay);
        const float laz  = flog_safe(az);
        const float xt   = __expf(p2 * lax);
        const float yt   = __expf(p2 * lay);
        const float zt   = __expf(p1 * laz);
        const float sum  = xt + yt;
        const float lsum = flog_safe(sum);
        const float F    = __expf(e_ratio * lsum) + zt;

        const float sx_sign = copysignf(1.f, qx);
        const float sy_sign = copysignf(1.f, qy);
        const float sz_sign = copysignf(1.f, qz);
        const float ps      = __expf((e_ratio - 1.f) * lsum);
        const float c       = 2.f * inv_e1;

        surf_gx = c * __frcp_rn(sq.sx) * sx_sign * ps * __expf((p2 - 1.f) * lax);
        surf_gy = c * __frcp_rn(sq.sy) * sy_sign * ps * __expf((p2 - 1.f) * lay);
        surf_gz = c * __frcp_rn(sq.sz) * sz_sign       * __expf((p1 - 1.f) * laz);

        const float df    = fmaf(surf_gx, lx, fmaf(surf_gy, ly, surf_gz * lz));
        const float delta = -(F - 1.f) / (df + 1e-8f);
        lambda += delta;
        if (fabsf(delta) < 1e-4f) break;
    }

    /* One extra forward pass at the final λ for an accurate surface gradient. */
    {
        const float qx = lambda * lx;
        const float qy = lambda * ly;
        const float qz = lambda * lz;

        const float ax = fmaxf(fabsf(qx) * __frcp_rn(sq.sx), 1e-9f);
        const float ay = fmaxf(fabsf(qy) * __frcp_rn(sq.sy), 1e-9f);
        const float az = fmaxf(fabsf(qz) * __frcp_rn(sq.sz), 1e-9f);

        const float lax  = flog_safe(ax);
        const float lay  = flog_safe(ay);
        const float laz  = flog_safe(az);
        const float sum  = __expf(p2 * lax) + __expf(p2 * lay);
        const float lsum = flog_safe(sum);
        const float ps   = __expf((e_ratio - 1.f) * lsum);
        const float c    = 2.f * inv_e1;

        surf_gx = c * __frcp_rn(sq.sx) * copysignf(1.f, qx) * ps * __expf((p2 - 1.f) * lax);
        surf_gy = c * __frcp_rn(sq.sy) * copysignf(1.f, qy) * ps * __expf((p2 - 1.f) * lay);
        surf_gz = c * __frcp_rn(sq.sz) * copysignf(1.f, qz)       * __expf((p1 - 1.f) * laz);
    }

    return lambda;
}


/* ═══════════════════════════ Core SDF ══════════════════════════════════════
 *
 * Returns the signed-distance for a sphere against a superquadric.
 *
 * Kernel sign convention (POSITIVE = outside / no collision):
 *   > 0  →  sphere outside SQ        (clearance ≈ value)
 *   < 0  →  sphere penetrates SQ     (penetration depth ≈ |value|)
 *
 * CuRobo convention is the opposite (positive = collision).  The C++ wrapper
 * negates the result before computing collision costs.
 *
 * Mathematics:
 *   F(p) = ((|x/sx|^{2/ε₂} + |y/sy|^{2/ε₂})^{ε₂/ε₁} + |z/sz|^{2/ε₁})
 *   F = 1 on the surface, > 1 outside, < 1 inside.
 *
 *   Outside (F ≥ 1): SDF ≈ (F − 1) / ‖∇F‖ − r_sphere   (Taubin approx.)
 *   Inside  (F < 1): SDF  = (1 − λ)·|p_local| − r_sphere  (Newton exact)
 * ═══════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__
float sq_sdf(
    const float px, const float py, const float pz, const float pr,
    const SQData& sq)
{
    // World-frame offset (keep for lower bound — rotation preserves distance)
    const float dx = px - sq.cx;
    const float dy = py - sq.cy;
    const float dz = pz - sq.cz;

    // Rotate into SQ local frame
    float lx, ly, lz;
    rotate_by_quat_inv(dx, dy, dz, sq.qw, sq.qx, sq.qy, sq.qz, lx, ly, lz);

    const float ax = fmaxf(fabsf(lx) * __frcp_rn(sq.sx), 1e-9f);
    const float ay = fmaxf(fabsf(ly) * __frcp_rn(sq.sy), 1e-9f);
    const float az = fmaxf(fabsf(lz) * __frcp_rn(sq.sz), 1e-9f);

    const float inv_e1  = __frcp_rn(sq.eps1);
    const float p1      = 2.f * inv_e1;
    const float p2      = 2.f * __frcp_rn(sq.eps2);
    const float e_ratio = sq.eps2 * inv_e1;

    const float lax  = flog_safe(ax);
    const float lay  = flog_safe(ay);
    const float laz  = flog_safe(az);
    const float xt   = __expf(p2 * lax);
    const float yt   = __expf(p2 * lay);
    const float zt   = __expf(p1 * laz);
    const float sum  = xt + yt;
    const float lsum = flog_safe(sum);
    const float F    = __expf(e_ratio * lsum) + zt;

    // ── Inside: Newton radial projection → exact signed distance ─────────
    if (F < 1.f) {
        const float p_len = sqrtf(fmaf(lx, lx, fmaf(ly, ly, lz * lz)));
        if (p_len < 1e-6f)
            return -fminf(sq.sx, fminf(sq.sy, sq.sz)) - pr;
        const float lambda = sq_newton_lambda(lx, ly, lz, sq, F);
        return fmaf(1.f - lambda, p_len, -pr);
    }

    // ── Outside: Taubin approximation + conservative lower bounds ─────────
    const float sx_sign = copysignf(1.f, lx);
    const float sy_sign = copysignf(1.f, ly);
    const float sz_sign = copysignf(1.f, lz);
    const float ps      = __expf((e_ratio - 1.f) * lsum);
    const float c       = 2.f * inv_e1;

    const float gx = c * __frcp_rn(sq.sx) * sx_sign * ps * __expf((p2 - 1.f) * lax);
    const float gy = c * __frcp_rn(sq.sy) * sy_sign * ps * __expf((p2 - 1.f) * lay);
    const float gz = c * __frcp_rn(sq.sz) * sz_sign       * __expf((p1 - 1.f) * laz);
    const float g2 = fmaf(gx, gx, fmaf(gy, gy, gz * gz));

    const float sdf_approx = fmaf(F - 1.f, rsqrtf(g2 + 1e-8f), -pr);

    // Coarse lower bound: world-frame distance to bounding sphere (rotation-invariant)
    const float r_outer = fmaxf(sq.sx, fmaxf(sq.sy, sq.sz));
    const float lb = sqrtf(fmaf(dx, dx, fmaf(dy, dy, dz * dz))) - r_outer - pr;

    // Tight lower bound: L2 distance to AABB in local SQ frame.
    const float dx_box = fmaxf(fabsf(lx) - sq.sx, 0.f);
    const float dy_box = fmaxf(fabsf(ly) - sq.sy, 0.f);
    const float dz_box = fmaxf(fabsf(lz) - sq.sz, 0.f);
    const float lb_box = sqrtf(fmaf(dx_box, dx_box, fmaf(dy_box, dy_box, dz_box * dz_box))) - pr;

    return fmaxf(sdf_approx, fmaxf(lb, lb_box));
}

/* ── sq_sdf_and_normal ───────────────────────────────────────────────────────
 *
 * Same as sq_sdf but also returns the world-frame unit outward normal
 * n̂ = R_local2world * (∇F_local / ‖∇F_local‖).
 *
 * Used by the analytical gradient kernels to replace numerical FD.
 * Sign convention for the returned SDF is identical to sq_sdf.
 */
__device__ __forceinline__
float sq_sdf_and_normal(
    const float px, const float py, const float pz, const float pr,
    const SQData& sq,
    float& nx_world, float& ny_world, float& nz_world)
{
    const float dx = px - sq.cx;
    const float dy = py - sq.cy;
    const float dz = pz - sq.cz;

    float lx, ly, lz;
    rotate_by_quat_inv(dx, dy, dz, sq.qw, sq.qx, sq.qy, sq.qz, lx, ly, lz);

    const float ax = fmaxf(fabsf(lx) * __frcp_rn(sq.sx), 1e-9f);
    const float ay = fmaxf(fabsf(ly) * __frcp_rn(sq.sy), 1e-9f);
    const float az = fmaxf(fabsf(lz) * __frcp_rn(sq.sz), 1e-9f);

    const float inv_e1  = __frcp_rn(sq.eps1);
    const float p1      = 2.f * inv_e1;
    const float p2      = 2.f * __frcp_rn(sq.eps2);
    const float e_ratio = sq.eps2 * inv_e1;

    const float lax  = flog_safe(ax);
    const float lay  = flog_safe(ay);
    const float laz  = flog_safe(az);
    const float xt   = __expf(p2 * lax);
    const float yt   = __expf(p2 * lay);
    const float zt   = __expf(p1 * laz);
    const float sum  = xt + yt;
    const float lsum = flog_safe(sum);
    const float F    = __expf(e_ratio * lsum) + zt;

    // ── Inside: Newton gives exact SDF and surface-point outward normal ───
    if (F < 1.f) {
        const float p_len = sqrtf(fmaf(lx, lx, fmaf(ly, ly, lz * lz)));
        if (p_len < 1e-6f) {
            // Near-origin fallback: point outward along local Z
            rotate_by_quat_fwd(0.f, 0.f, 1.f,
                               sq.qw, sq.qx, sq.qy, sq.qz,
                               nx_world, ny_world, nz_world);
            return -fminf(sq.sx, fminf(sq.sy, sq.sz)) - pr;
        }
        float sgx, sgy, sgz;
        const float lambda = sq_newton_lambda_and_surf_grad(lx, ly, lz, sq,
                                                             sgx, sgy, sgz, F);
        const float inv_sg = rsqrtf(fmaf(sgx, sgx, fmaf(sgy, sgy, sgz * sgz)) + 1e-8f);
        rotate_by_quat_fwd(sgx * inv_sg, sgy * inv_sg, sgz * inv_sg,
                           sq.qw, sq.qx, sq.qy, sq.qz,
                           nx_world, ny_world, nz_world);
        return fmaf(1.f - lambda, p_len, -pr);
    }

    // ── Outside: Taubin approximation + normal at query point ─────────────
    const float sx_sign = copysignf(1.f, lx);
    const float sy_sign = copysignf(1.f, ly);
    const float sz_sign = copysignf(1.f, lz);
    const float ps      = __expf((e_ratio - 1.f) * lsum);
    const float c       = 2.f * inv_e1;

    const float gx = c * __frcp_rn(sq.sx) * sx_sign * ps * __expf((p2 - 1.f) * lax);
    const float gy = c * __frcp_rn(sq.sy) * sy_sign * ps * __expf((p2 - 1.f) * lay);
    const float gz = c * __frcp_rn(sq.sz) * sz_sign       * __expf((p1 - 1.f) * laz);
    const float g2 = fmaf(gx, gx, fmaf(gy, gy, gz * gz));

    const float inv_g      = rsqrtf(g2 + 1e-8f);
    const float sdf_approx = fmaf(F - 1.f, inv_g, -pr);

    // Lower bounds (identical to sq_sdf outside path)
    const float r_outer = fmaxf(sq.sx, fmaxf(sq.sy, sq.sz));
    const float lb      = sqrtf(fmaf(dx, dx, fmaf(dy, dy, dz * dz))) - r_outer - pr;
    const float dx_box  = fmaxf(fabsf(lx) - sq.sx, 0.f);
    const float dy_box  = fmaxf(fabsf(ly) - sq.sy, 0.f);
    const float dz_box  = fmaxf(fabsf(lz) - sq.sz, 0.f);
    const float lb_box  = sqrtf(fmaf(dx_box, dx_box, fmaf(dy_box, dy_box, dz_box * dz_box))) - pr;

    const float lnx = gx * inv_g;
    const float lny = gy * inv_g;
    const float lnz = gz * inv_g;
    rotate_by_quat_fwd(lnx, lny, lnz,
                       sq.qw, sq.qx, sq.qy, sq.qz,
                       nx_world, ny_world, nz_world);

    return fmaxf(sdf_approx, fmaxf(lb, lb_box));
}


/* ═══════════════════════════ CUDA kernels ══════════════════════════════════ */

static constexpr int BLOCK          = 128;  // threads per block
static constexpr int SQ_TILE        = 64;   // SQs per tile for min-distance kernels
static constexpr int WARP_SZ        = 32;   // warp size (CUDA architecture constant)
static constexpr int WARPS_PER_BLOCK = BLOCK / WARP_SZ;  // 4
static constexpr float MIN_RADIUS   = 1e-2f;  // Avoid degenerate SQ axes

/* ── Kernel A: minimum raw distance over all obstacles ──────────────────────
 *
 * Output convention (same as sq_sdf): positive = outside, negative = inside.
 * The wrapper negates this to obtain CuRobo's penetration-positive SDF.
 */
__global__
void sphere_sq_min_kernel(
    const float*  __restrict__ spheres,   // [n_spheres, 4]: x,y,z,r  row-major
    const SQData* __restrict__ sq_arr,    // [n_obs]
    float*        __restrict__ out_dist,  // [n_spheres]
    const int     n_spheres,
    const int     n_obs)
{
    __shared__ SQData sh[SQ_TILE];

    const int tid   = threadIdx.x;
    const int gid   = blockIdx.x * BLOCK + tid;
    const bool valid = (gid < n_spheres);

    /* 128-bit coalesced load of (x,y,z,r) */
    float px = 0.f, py = 0.f, pz = 0.f, pr = 0.f;
    if (valid) {
        const float4 s = __ldg(reinterpret_cast<const float4*>(spheres) + gid);
        if (s.w < 0.f) { out_dist[gid] = 1e10f; return; }  // disabled sphere
        px = s.x; py = s.y; pz = s.z; pr = s.w;
    }

    float min_d = 1e10f;

    /* Tile loop: load SQ_TILE descriptors into shared memory per iteration */
    for (int base = 0; base < n_obs; base += SQ_TILE) {
        /* Cooperative load — only threads 0..(SQ_TILE-1) fetch data */
        if (tid < SQ_TILE) {
            const int load_i = base + tid;
            if (load_i < n_obs)
                sh[tid] = sq_arr[load_i];
        }
        __syncthreads();

        if (valid) {
            const int tile_end = min(SQ_TILE, n_obs - base);
            for (int j = 0; j < tile_end; ++j) {
                if (sh[j]._pad == 0.f) continue;
                min_d = fminf(min_d, sq_sdf(px, py, pz, pr, sh[j]));
            }
        }
        __syncthreads();
    }

    if (valid)
        out_dist[gid] = min_d;
}

/* ── Kernel A2: minimum raw distance + world-frame unit outward normal ───────
 *
 * Extends sphere_sq_min_kernel to also output the analytical outward normal
 * n̂_world = R_local2world * (∇F/‖∇F‖) for the closest obstacle.
 * Used by the analytical gradient path to replace 6 numerical FD launches.
 *
 * out_grad layout: float4 per sphere, xyz = world-frame normal, w = 0.
 * Sign convention: normal points outward (away from SQ interior).
 */
__global__
void sphere_sq_min_and_grad_kernel(
    const float*  __restrict__ spheres,   // [n_spheres, 4]
    const SQData* __restrict__ sq_arr,    // [n_obs]
    float*        __restrict__ out_dist,  // [n_spheres]
    float4*       __restrict__ out_grad,  // [n_spheres]  world-frame normal
    const int     n_spheres,
    const int     n_obs)
{
    __shared__ SQData sh[SQ_TILE];

    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * BLOCK + tid;
    const bool valid = (gid < n_spheres);

    float px = 0.f, py = 0.f, pz = 0.f, pr = 0.f;
    if (valid) {
        const float4 s = __ldg(reinterpret_cast<const float4*>(spheres) + gid);
        if (s.w < 0.f) {
            out_dist[gid] = 1e10f;
            out_grad[gid] = make_float4(0.f, 0.f, 0.f, 0.f);
            return;
        }
        px = s.x; py = s.y; pz = s.z; pr = s.w;
    }

    float min_d   = 1e10f;
    float best_nx = 0.f, best_ny = 0.f, best_nz = 1.f;  // world-frame normal

    for (int base = 0; base < n_obs; base += SQ_TILE) {
        if (tid < SQ_TILE) {
            const int load_i = base + tid;
            if (load_i < n_obs)
                sh[tid] = sq_arr[load_i];
        }
        __syncthreads();

        if (valid) {
            const int tile_end = min(SQ_TILE, n_obs - base);
            for (int j = 0; j < tile_end; ++j) {
                if (sh[j]._pad == 0.f) continue;
                float wnx, wny, wnz;
                const float d = sq_sdf_and_normal(px, py, pz, pr, sh[j],
                                                  wnx, wny, wnz);
                if (d < min_d) {
                    min_d    = d;
                    best_nx  = wnx;
                    best_ny  = wny;
                    best_nz  = wnz;
                }
            }
        }
        __syncthreads();
    }

    if (valid) {
        out_dist[gid] = min_d;
        out_grad[gid] = make_float4(best_nx, best_ny, best_nz, 0.f);
    }
}

/* ── Kernel B: sum of smoothed collision costs over all obstacles ────────────
 *
 * Warp-per-sphere design: each warp of WARP_SZ=32 threads handles ONE sphere.
 * Lanes within the warp each evaluate a different subset of SQs (stride=WARP_SZ),
 * then the partial costs are summed via warp-shuffle reduction. This exposes
 * n_obs-way parallelism instead of serialising SQs per thread, which reduces
 * the sequential loop from n_obs iterations to ceil(n_obs/WARP_SZ) iterations.
 *
 * All BLOCK=128 threads cooperatively load BLOCK SQs into shared memory
 * (vs. only 64 of 128 threads in the old SQ_TILE=64 scheme).
 *
 * Launch: grid = ceil(n_spheres / WARPS_PER_BLOCK), block = BLOCK.
 */
__global__
void sphere_sq_sum_cost_kernel(
    const float*  __restrict__ spheres,   // [n_spheres, 4]
    const SQData* __restrict__ sq_arr,    // [n_obs]
    float*        __restrict__ out_cost,  // [n_spheres]  (unweighted)
    const int     n_spheres,
    const int     n_obs,
    const float   act_dist)
{
    __shared__ SQData sh[BLOCK];  // 128 × 64 bytes = 8 KB

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane    = tid & (WARP_SZ - 1);
    const int gid     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool valid  = (gid < n_spheres);

    float px = 0.f, py = 0.f, pz = 0.f, pr = -1.f;
    if (valid) {
        const float4 s = __ldg(reinterpret_cast<const float4*>(spheres) + gid);
        px = s.x; py = s.y; pz = s.z; pr = s.w;
    }

    const bool active = valid && (pr >= 0.f);
    float partial = 0.f;

    for (int base = 0; base < n_obs; base += BLOCK) {
        /* All BLOCK threads load BLOCK SQ descriptors cooperatively */
        const int load_i = base + tid;
        if (load_i < n_obs)
            sh[tid] = sq_arr[load_i];
        __syncthreads();

        if (active) {
            const int tile_end = min(BLOCK, n_obs - base);
            /* Each lane handles SQs: lane, lane+WARP_SZ, lane+2*WARP_SZ, ... */
            for (int j = lane; j < tile_end; j += WARP_SZ) {
                if (sh[j]._pad == 0.f) continue;
                if (sq_aabb_miss(px, py, pz, pr, sh[j])) continue;
                const float sdf = -sq_sdf(px, py, pz, pr, sh[j]);
                if (sdf > 0.f) {
                    partial += (act_dist > 0.f)
                        ? ((sdf > act_dist)
                            ? sdf - 0.5f * act_dist
                            : (0.5f / act_dist) * sdf * sdf)
                        : sdf;
                }
            }
        }
        __syncthreads();
    }

    /* Warp-level reduction: sum partial costs from all 32 lanes */
    partial += __shfl_down_sync(0xffffffff, partial, 16);
    partial += __shfl_down_sync(0xffffffff, partial,  8);
    partial += __shfl_down_sync(0xffffffff, partial,  4);
    partial += __shfl_down_sync(0xffffffff, partial,  2);
    partial += __shfl_down_sync(0xffffffff, partial,  1);

    if (lane == 0 && valid)
        out_cost[gid] = partial;
}

/* ── Kernel B2: sum-of-costs + accumulated analytical gradient ───────────────
 *
 * Same warp-per-sphere design as sphere_sq_sum_cost_kernel.
 * Each lane also accumulates a partial normal vector (sum_gnx/y/z), which is
 * reduced across the warp with four independent __shfl_down_sync chains.
 *
 * cost'(sdf) = 0          if sdf ≤ 0
 *            = sdf/act    if 0 < sdf ≤ act_dist   (quadratic region)
 *            = 1          if sdf > act_dist        (linear region)
 *
 * Launch: grid = ceil(n_spheres / WARPS_PER_BLOCK), block = BLOCK.
 */
__global__
void sphere_sq_sum_cost_and_grad_kernel(
    const float*  __restrict__ spheres,   // [n_spheres, 4]
    const SQData* __restrict__ sq_arr,    // [n_obs]
    float*        __restrict__ out_cost,  // [n_spheres]  (unweighted)
    float4*       __restrict__ out_grad,  // [n_spheres]  accumulated weighted normals
    const int     n_spheres,
    const int     n_obs,
    const float   act_dist)
{
    __shared__ SQData sh[BLOCK];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane    = tid & (WARP_SZ - 1);
    const int gid     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const bool valid  = (gid < n_spheres);

    float px = 0.f, py = 0.f, pz = 0.f, pr = -1.f;
    if (valid) {
        const float4 s = __ldg(reinterpret_cast<const float4*>(spheres) + gid);
        px = s.x; py = s.y; pz = s.z; pr = s.w;
    }

    const bool active = valid && (pr >= 0.f);
    float partial = 0.f;
    float sum_gnx = 0.f, sum_gny = 0.f, sum_gnz = 0.f;

    for (int base = 0; base < n_obs; base += BLOCK) {
        const int load_i = base + tid;
        if (load_i < n_obs)
            sh[tid] = sq_arr[load_i];
        __syncthreads();

        if (active) {
            const int tile_end = min(BLOCK, n_obs - base);
            for (int j = lane; j < tile_end; j += WARP_SZ) {
                if (sh[j]._pad == 0.f) continue;
                if (sq_aabb_miss(px, py, pz, pr, sh[j])) continue;
                float wnx, wny, wnz;
                const float sdf = -sq_sdf_and_normal(px, py, pz, pr, sh[j],
                                                     wnx, wny, wnz);
                if (sdf > 0.f) {
                    float cost_d;
                    if (act_dist > 0.f) {
                        if (sdf > act_dist) {
                            partial += sdf - 0.5f * act_dist;
                            cost_d   = 1.f;
                        } else {
                            partial += (0.5f / act_dist) * sdf * sdf;
                            cost_d   = sdf / act_dist;
                        }
                    } else {
                        partial += sdf;
                        cost_d   = 1.f;
                    }
                    sum_gnx += cost_d * wnx;
                    sum_gny += cost_d * wny;
                    sum_gnz += cost_d * wnz;
                }
            }
        }
        __syncthreads();
    }

    /* Warp-level reduction across all four accumulators */
    partial  += __shfl_down_sync(0xffffffff, partial,  16);
    partial  += __shfl_down_sync(0xffffffff, partial,   8);
    partial  += __shfl_down_sync(0xffffffff, partial,   4);
    partial  += __shfl_down_sync(0xffffffff, partial,   2);
    partial  += __shfl_down_sync(0xffffffff, partial,   1);

    sum_gnx  += __shfl_down_sync(0xffffffff, sum_gnx,  16);
    sum_gnx  += __shfl_down_sync(0xffffffff, sum_gnx,   8);
    sum_gnx  += __shfl_down_sync(0xffffffff, sum_gnx,   4);
    sum_gnx  += __shfl_down_sync(0xffffffff, sum_gnx,   2);
    sum_gnx  += __shfl_down_sync(0xffffffff, sum_gnx,   1);

    sum_gny  += __shfl_down_sync(0xffffffff, sum_gny,  16);
    sum_gny  += __shfl_down_sync(0xffffffff, sum_gny,   8);
    sum_gny  += __shfl_down_sync(0xffffffff, sum_gny,   4);
    sum_gny  += __shfl_down_sync(0xffffffff, sum_gny,   2);
    sum_gny  += __shfl_down_sync(0xffffffff, sum_gny,   1);

    sum_gnz  += __shfl_down_sync(0xffffffff, sum_gnz,  16);
    sum_gnz  += __shfl_down_sync(0xffffffff, sum_gnz,   8);
    sum_gnz  += __shfl_down_sync(0xffffffff, sum_gnz,   4);
    sum_gnz  += __shfl_down_sync(0xffffffff, sum_gnz,   2);
    sum_gnz  += __shfl_down_sync(0xffffffff, sum_gnz,   1);

    if (lane == 0 && valid) {
        out_cost[gid] = partial;
        out_grad[gid] = make_float4(sum_gnx, sum_gny, sum_gnz, 0.f);
    }
}

/* ═══════════════════════════ C++ helpers ═══════════════════════════════════ */

namespace {

static int clamp_env(int e, int maxe)
{
    return std::max(0, std::min(e, maxe - 1));
}

/* ── pack_env_sq ─────────────────────────────────────────────────────────────
 *
 * Repack SQ parameters for one environment into the SQData layout required by
 * the CUDA kernels.  All max_nobs slots are always packed (fixed output size),
 * which avoids mask.nonzero() and makes this function compatible with CUDA
 * graph capture.  Disabled SQs get _pad=0.0 (col 15) so kernels can skip them
 * with a single float comparison before doing any expensive work.
 *
 * Input layout:  [sx,sy,sz, eps1,eps2, cx,cy,cz, qx,qy,qz,qw]
 * Output layout: cx,cy,cz, sx,sy,sz, eps1,eps2, qw,qx,qy,qz, hx,hy,hz, _pad
 *
 * Returns a contiguous float32 GPU tensor of shape [max_nobs, 16].
 */
torch::Tensor pack_env_sq(
    const torch::Tensor& sq_params,
    const torch::Tensor& enabled_mask,
    const int env_idx)
{
    auto dev_opts = torch::TensorOptions()
                        .dtype(torch::kFloat)
                        .device(sq_params.device());

    const int max_nobs = (int)sq_params.size(1);
    if (max_nobs == 0)
        return torch::empty({0, 16}, dev_opts);

    const auto raw  = sq_params[env_idx].to(torch::kFloat);        // [max_nobs, 12]
    const auto mask = enabled_mask[env_idx].to(torch::kFloat)      // [max_nobs, 1]
                          .unsqueeze(1);

    auto out = torch::zeros({max_nobs, 16}, dev_opts);

    // cx, cy, cz — zero for disabled SQs
    out.slice(1, 0, 3).copy_(raw.slice(1, 5, 8) * mask);
    // sx, sy, sz — clamped, then zeroed for disabled SQs
    out.slice(1, 3, 6).copy_(
        torch::clamp(torch::abs(raw.slice(1, 0, 3)), MIN_RADIUS) * mask);
    // eps1, eps2
    const auto mask1 = mask.select(1, 0);
    out.select(1, 6).copy_(
        torch::clamp(torch::abs(raw.select(1, 3)), 0.05f, 4.0f) * mask1);
    out.select(1, 7).copy_(
        torch::clamp(torch::abs(raw.select(1, 4)), 0.05f, 4.0f) * mask1);

    // Quaternion: normalise and reorder [qx,qy,qz,qw] → [qw,qx,qy,qz]
    auto q      = raw.slice(1, 8, 12);
    auto q_norm = torch::clamp(torch::norm(q, 2, 1, true), 1e-6f);
    auto q_n    = (q / q_norm * mask).contiguous();
    out.select(1,  8).copy_(q_n.select(1, 3));       // qw
    out.slice(1,  9, 12).copy_(q_n.slice(1, 0, 3));  // qx,qy,qz

    // World-frame AABB half-extents: h_i = Σ_j |R_ij| · s_j
    const auto qw  = out.select(1,  8);
    const auto qx  = out.select(1,  9);
    const auto qy  = out.select(1, 10);
    const auto qz  = out.select(1, 11);
    const auto sx  = out.select(1,  3);
    const auto sy  = out.select(1,  4);
    const auto sz  = out.select(1,  5);
    const auto qx2 = qx * qx, qy2 = qy * qy, qz2 = qz * qz;
    out.select(1, 12).copy_(
        (1.f - 2.f * (qy2 + qz2)).abs() * sx +
        (2.f * (qx * qy - qw * qz)).abs() * sy +
        (2.f * (qx * qz + qw * qy)).abs() * sz);   // hx
    out.select(1, 13).copy_(
        (2.f * (qx * qy + qw * qz)).abs() * sx +
        (1.f - 2.f * (qx2 + qz2)).abs() * sy +
        (2.f * (qy * qz - qw * qx)).abs() * sz);   // hy
    out.select(1, 14).copy_(
        (2.f * (qx * qz - qw * qy)).abs() * sx +
        (2.f * (qy * qz + qw * qx)).abs() * sy +
        (1.f - 2.f * (qx2 + qy2)).abs() * sz);     // hz
    out.select(1, 15).copy_(mask1);                  // _pad = enabled flag

    return out.contiguous();
}

/* ── Smooth quadratic–linear collision cost (tensor form) ─────────────────── */
torch::Tensor sdf_to_collision_cost(
    const torch::Tensor& sdf,
    const torch::Tensor& act_dist)
{
    const auto pos    = torch::relu(sdf);
    const auto asafe  = torch::clamp(act_dist, 1.0e-12);
    const auto lin    = pos - 0.5 * asafe;
    const auto quad   = pos * pos * (0.5 / asafe);
    const auto smooth = torch::where(pos > asafe, lin, quad);
    const auto ha     = (act_dist > 0.).to(pos.scalar_type());
    return ha * smooth + (1. - ha) * pos;
}

/* ══════════════════════════════════════════════════════════════════════════
 * evaluate_all_sq
 *
 * Evaluates ALL enabled obstacles for ONE environment against all q_count
 * query spheres in a single kernel launch.
 *
 * Returns a [q_count] float tensor with:
 *   - For spheres in env_idx:  cost/SDF value
 *   - For spheres not in env_idx: 0 (non-ESDF) or -1e6 (ESDF)
 *
 * Key fix over original: sq_params are packed into [n_obs, 8] once, and the
 * kernel iterates over all n_obs obstacles internally — no per-obstacle loop.
 * ══════════════════════════════════════════════════════════════════════════ */
torch::Tensor evaluate_all_sq(
    const torch::Tensor& query_spheres,   // [q, 4]  float32 contiguous
    const torch::Tensor& sq_params,       // [nenv, maxobs, 8]
    const torch::Tensor& enabled_mask,    // [nenv, maxobs] bool
    const torch::Tensor& query_env_idx,   // [q] int64
    const int   env_idx,
    const torch::Tensor& weight,
    const torch::Tensor& act_dist,
    const bool  sum_collisions,
    const bool  compute_esdf,
    cudaStream_t stream)
{
    const int64_t q   = query_spheres.size(0);
    const auto    opt = query_spheres.options().dtype(torch::kFloat);

    if (q == 0)
        return torch::empty({0}, opt);

    const int cenv      = clamp_env(env_idx, (int)sq_params.size(0));
    const auto env_mask = (query_env_idx == (int64_t)cenv);

    auto values = compute_esdf ? torch::full({q}, -1e6f, opt)
                               : torch::zeros({q}, opt);

    /* ── Pack SQ descriptors for this environment (fixed [max_nobs,16] size) ── */
    const auto sq_packed = pack_env_sq(sq_params, enabled_mask, cenv);
    const int n_obs = (int)sq_packed.size(0);   // = max_nobs; disabled SQs have _pad=0
    if (n_obs == 0)
        return values;

    auto raw = torch::empty({q}, opt);    // kernel output buffer

    /* min-distance kernels: 1 thread per sphere */
    const int blocks       = ((int)q + BLOCK          - 1) / BLOCK;
    /* cost kernels: 1 warp (WARP_SZ threads) per sphere */
    const int cost_blocks  = ((int)q + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const auto* sq_ptr = reinterpret_cast<const SQData*>(sq_packed.data_ptr<float>());
    const float* sp    = query_spheres.data_ptr<float>();

    /* ── Extract scalar parameters ──────────────────────────────────── */
    const float ad = act_dist.numel() > 0
                     ? act_dist.flatten().select(0, 0).item<float>() : 0.f;
    const float wt = weight.numel() > 0
                     ? weight.flatten().select(0, 0).item<float>()   : 1.f;

    /* ════ Sum-collisions path ════════════════════════════════════════ */
    if (sum_collisions && !compute_esdf) {
        sphere_sq_sum_cost_kernel<<<cost_blocks, BLOCK, 0, stream>>>(
            sp, sq_ptr, raw.data_ptr<float>(), (int)q, n_obs, ad);

        /* raw = unweighted sum-of-costs for all spheres; zero out off-env */
        values = values + raw * env_mask.to(torch::kFloat) * wt;
        return values;
    }

    /* ════ Min-distance path (ESDF and max-cost) ══════════════════════ */
    sphere_sq_min_kernel<<<blocks, BLOCK, 0, stream>>>(
        sp, sq_ptr, raw.data_ptr<float>(), (int)q, n_obs);

    const auto sdf = -raw;   // negate: positive = penetrating (CuRobo convention)

    if (compute_esdf) {
        /* ESDF: maximum penetration depth across all obstacles */
        values = torch::where(env_mask, torch::maximum(values, sdf), values);
    } else {
        /* Max-cost: cost of the most-penetrating obstacle */
        const auto cost = sdf_to_collision_cost(sdf, act_dist) * wt;
        values = torch::where(env_mask, torch::maximum(values, cost), values);
    }
    return values;
}

/* ══════════════════════════════════════════════════════════════════════════
 * evaluate_swept_sq
 *
 * Swept-sphere variant: samples sphere positions along the motion segment
 * [centre, next_centre] at (sweep_steps+1) uniformly-spaced alphas and
 * returns the element-wise maximum cost/SDF over all samples.
 * ══════════════════════════════════════════════════════════════════════════ */
torch::Tensor evaluate_swept_sq(
    const torch::Tensor& query_spheres,   // [batch, horizon, n_sph, 4]
    const torch::Tensor& sq_params,
    const torch::Tensor& enabled_mask,
    const torch::Tensor& query_env_idx,   // [total_queries] int64
    const torch::Tensor& weight,
    const torch::Tensor& act_dist,
    const torch::Tensor& speed_dt,
    const int   sweep_steps,
    const bool  enable_speed_metric,
    const bool  sum_collisions,
    const bool  compute_esdf,
    cudaStream_t stream)
{
    const int64_t B  = query_spheres.size(0);
    const int64_t H  = query_spheres.size(1);
    const int64_t S  = query_spheres.size(2);
    const int64_t T  = B * H * S;
    const auto    opt = query_spheres.options().dtype(torch::kFloat);

    if (T == 0)
        return torch::empty({0}, opt);

    const auto qv      = query_spheres.view({B, H, S, 4}).contiguous();
    const auto centres = qv.slice(3, 0, 3).contiguous();
    const auto radii   = torch::abs(qv.select(3, 3)).contiguous();

    /* next_centres: shifted by 1 along horizon; last step repeats final */
    const auto next_c = torch::cat(
        {centres.slice(1, 1, H), centres.slice(1, H - 1, H)}, 1).contiguous();

    auto agg = compute_esdf ? torch::full({T}, -1e6f, opt)
                            : torch::zeros({T}, opt);

    const int eff_steps = std::max(sweep_steps, 1);
    const int num_envs  = std::max((int)sq_params.size(0), 1);

    /* ── Speed-scaling factor  s = clamp(1 + ‖Δc‖/dt, 1, 50) ────────── */
    auto speed_scale = torch::ones({T}, opt);
    if (enable_speed_metric && !compute_esdf) {
        const auto delta   = (next_c - centres).contiguous();
        const auto step_len = torch::sqrt(torch::sum(delta * delta, 3)).view({T});
        const float dt = (speed_dt.numel() > 0)
            ? std::max(torch::abs(speed_dt.flatten().select(0, 0)).item<float>(), 1e-6f)
            : 1.f;
        speed_scale = torch::clamp(1.f + step_len / dt, 1.f, 50.f);
    }

    const auto c_flat = centres.view({T, 3}).contiguous();
    const auto d_flat = (next_c - centres).view({T, 3}).contiguous();
    const auto r_flat = radii.view({T, 1}).contiguous();

    for (int step = 0; step <= eff_steps; ++step) {
        const float alpha   = (float)step / (float)eff_steps;
        const auto  samp_c  = (c_flat + d_flat * alpha).contiguous();
        const auto  samp_sp = torch::cat({samp_c, r_flat}, 1).contiguous();

        auto sv = compute_esdf ? torch::full({T}, -1e6f, opt)
                               : torch::zeros({T}, opt);

        for (int e = 0; e < num_envs; ++e) {
            auto ev = evaluate_all_sq(
                samp_sp, sq_params, enabled_mask, query_env_idx,
                e, weight, act_dist, sum_collisions, compute_esdf, stream);

            if (compute_esdf)
                sv = torch::where(query_env_idx == (int64_t)e, ev, sv);
            else
                sv = sv + ev;
        }

        if (enable_speed_metric && !compute_esdf)
            sv = sv * speed_scale;

        agg = torch::maximum(agg, sv);
    }
    return agg;
}

/* ══════════════════════════════════════════════════════════════════════════
 * evaluate_all_sq_grad
 *
 * Computes the analytical gradient of the collision cost (or ESDF) with
 * respect to sphere positions for ONE environment.
 *
 * Returns a [q, 4] tensor where [:3] = ∂cost/∂p_world and [3] = 0.
 * Off-environment spheres have gradient zero (env mask applied internally).
 *
 * Sign convention (matching numerical FD):
 *   ∂cost/∂p = -wt * cost'(sdf_curobo) * n̂_world
 *
 * where n̂_world is the unit outward normal from sq_sdf_and_normal and
 * sdf_curobo = -d_raw (positive inside, i.e. collision).
 * ══════════════════════════════════════════════════════════════════════════ */
torch::Tensor evaluate_all_sq_grad(
    const torch::Tensor& query_spheres,   // [q, 4]  float32 contiguous
    const torch::Tensor& sq_params,       // [nenv, maxobs, 12]
    const torch::Tensor& enabled_mask,    // [nenv, maxobs] bool
    const torch::Tensor& query_env_idx,   // [q] int64
    const int   env_idx,
    const torch::Tensor& weight,
    const torch::Tensor& act_dist,
    const bool  sum_collisions,
    const bool  compute_esdf,
    cudaStream_t stream)
{
    const int64_t q   = query_spheres.size(0);
    const auto    opt = query_spheres.options().dtype(torch::kFloat);

    auto grad = torch::zeros({q, 4}, opt);
    if (q == 0) return grad;

    const int  cenv       = clamp_env(env_idx, (int)sq_params.size(0));
    const auto env_mask_f = (query_env_idx == (int64_t)cenv).to(opt.dtype());

    const auto sq_packed = pack_env_sq(sq_params, enabled_mask, cenv);
    const int  n_obs     = (int)sq_packed.size(0);   // = max_nobs; _pad flags disabled SQs
    if (n_obs == 0) return grad;

    const int    blocks      = ((int)q + BLOCK          - 1) / BLOCK;
    const int    cost_blocks = ((int)q + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const auto*  sq_ptr = reinterpret_cast<const SQData*>(sq_packed.data_ptr<float>());
    const float* sp     = query_spheres.data_ptr<float>();

    const float ad = act_dist.numel() > 0
                     ? act_dist.flatten().select(0, 0).item<float>() : 0.f;
    const float wt = weight.numel() > 0
                     ? weight.flatten().select(0, 0).item<float>()   : 1.f;

    auto raw_dist = torch::empty({q}, opt);
    auto raw_grad = torch::empty({q, 4}, opt);   // float4-aligned via .contiguous()

    if (sum_collisions && !compute_esdf) {
        /* ── Sum path: accumulate Σ cost'_i * n̂_i ─────────────────────── */
        sphere_sq_sum_cost_and_grad_kernel<<<cost_blocks, BLOCK, 0, stream>>>(
            sp, sq_ptr,
            raw_dist.data_ptr<float>(),
            reinterpret_cast<float4*>(raw_grad.data_ptr<float>()),
            (int)q, n_obs, ad);

        /* ∂cost/∂p = -wt * Σ cost'_i * n̂_i  (kernel stores positive sum) */
        grad = (-wt) * raw_grad * env_mask_f.unsqueeze(1);
    } else {
        /* ── Min-distance path ──────────────────────────────────────────── */
        sphere_sq_min_and_grad_kernel<<<blocks, BLOCK, 0, stream>>>(
            sp, sq_ptr,
            raw_dist.data_ptr<float>(),
            reinterpret_cast<float4*>(raw_grad.data_ptr<float>()),
            (int)q, n_obs);

        /* sdf_curobo = -raw_dist  (positive = collision) */
        const auto sdf_curobo = -raw_dist;

        torch::Tensor cost_d;
        if (compute_esdf) {
            /* ESDF: ∂sdf_curobo/∂p = -n̂  → gradient = -n̂ (no weight) */
            cost_d = torch::ones({q}, opt);
        } else if (ad > 0.f) {
            /* Quadratic–linear cost derivative */
            const auto pos = torch::relu(sdf_curobo);
            const auto lin = torch::ones_like(pos);
            const auto qua = pos * (1.f / ad);
            cost_d = torch::where(pos > ad, lin, qua);
            cost_d = torch::where(sdf_curobo > 0.f, cost_d,
                                  torch::zeros_like(cost_d));
        } else {
            /* No activation: cost = relu(sdf), cost' = (sdf > 0) */
            cost_d = (sdf_curobo > 0.f).to(opt.dtype());
        }

        /* ∂cost/∂p = -wt * cost'(sdf) * n̂_world */
        const auto scale = (-wt) * cost_d;
        grad = raw_grad * scale.unsqueeze(1) * env_mask_f.unsqueeze(1);
    }

    return grad;
}

} // anonymous namespace

/* ═══════════════════════════════════════════════════════════════════════════
 * Batch environment kernels
 *
 * These kernels eliminate per-environment loops by handling all environments
 * in a single launch, matching the OBB kernel architecture for efficiency.
 *
 * Each thread block handles one sphere; accesses all environments for that
 * sphere via env_query_idx[] lookups, accumulating costs across all packed SQ
 * environments into a single output buffer.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Batch environment min-distance kernel (max-cost and ESDF paths) ─────── */
template <bool COMPUTE_ESDF>
__global__
void sphere_sq_batch_env_min_kernel(
    const float*  __restrict__ spheres,         // [batch, horizon, n_spheres, 4]
    float*        __restrict__ out_dist,        // [batch, horizon, n_spheres]
    const SQData* __restrict__ all_sq_packed,   // [num_envs * max_nobs]
    const int*    __restrict__ sq_env_starts,   // [num_envs]   offsets into all_sq_packed
    const int*    __restrict__ sq_env_counts,   // [num_envs]   valid SQ count per env
    const int32_t*__restrict__ env_query_idx,   // [batch]      env index per batch entry
    const float*  __restrict__ weight,
    const float*  __restrict__ activation_distance,
    const int     batch_size,
    const int     horizon,
    const int     n_spheres,
    const int     num_envs)
{
    __shared__ SQData sh[SQ_TILE];

    const int tid   = threadIdx.x;
    const int gid   = blockIdx.x * BLOCK + tid;
    const int total = batch_size * horizon * n_spheres;
    
    if (gid >= total) return;

    /* Decompose global index into batch, horizon, sphere */
    const int b_idx = gid / (horizon * n_spheres);
    const int h_idx = (gid - b_idx * (horizon * n_spheres)) / n_spheres;
    const int s_idx = gid - b_idx * horizon * n_spheres - h_idx * n_spheres;

    /* Load sphere position and radius */
    const float4 s = __ldg(reinterpret_cast<const float4*>(spheres) + gid);
    const float px = s.x, py = s.y, pz = s.z, pr = s.w;
    
    if (pr < 0.f) {
        out_dist[gid] = COMPUTE_ESDF ? -1e6f : 0.f;
        return;
    }

    /* Determine environment for this batch entry */
    const int env_idx = env_query_idx != nullptr ? env_query_idx[b_idx] : 0;
    const int env_idx_clamped = clamp_env(env_idx, num_envs);
    
    const int sq_start = sq_env_starts[env_idx_clamped];
    const int sq_count = sq_env_counts[env_idx_clamped];
    
    float min_dist = 1e10f;

    /* Tile loop over all SQs in this environment */
    for (int base = 0; base < sq_count; base += SQ_TILE) {
        if (tid < SQ_TILE) {
            const int load_i = sq_start + base + tid;
            if (load_i < sq_start + sq_count)
                sh[tid] = all_sq_packed[load_i];
        }
        __syncthreads();

        const int tile_end = min(SQ_TILE, sq_count - base);
        for (int j = 0; j < tile_end; ++j) {
            if (sh[j]._pad == 0.f) continue;
            min_dist = fminf(min_dist, sq_sdf(px, py, pz, pr, sh[j]));
        }
        __syncthreads();
    }

    /* Convert to CuRobo convention (negate) */
    const float sdf_curobo = -min_dist;
    
    const float ad = activation_distance != nullptr
                     ? activation_distance[0] : 0.f;
    const float wt = weight != nullptr ? weight[0] : 1.f;

    float out_value;
    if (COMPUTE_ESDF) {
        out_value = sdf_curobo;
    } else {
        /* Calculate collision cost from SDF */
        const float pos = fmaxf(sdf_curobo, 0.f);
        float cost;
        if (ad > 0.f) {
            cost = (pos > ad) ? (pos - 0.5f * ad) : ((0.5f / ad) * pos * pos);
        } else {
            cost = pos;
        }
        out_value = cost * wt;
    }
    
    out_dist[gid] = out_value;
}

/* ── Batch environment sum-cost kernel (accumulate cost over all SQs) ───── */
__global__
void sphere_sq_batch_env_sum_kernel(
    const float*  __restrict__ spheres,         // [batch, horizon, n_spheres, 4]
    float*        __restrict__ out_dist,        // [batch, horizon, n_spheres]
    const SQData* __restrict__ all_sq_packed,   // [num_envs * max_nobs]
    const int*    __restrict__ sq_env_starts,   // [num_envs]   offsets
    const int*    __restrict__ sq_env_counts,   // [num_envs]   counts
    const int32_t*__restrict__ env_query_idx,   // [batch]
    const float*  __restrict__ weight,
    const float*  __restrict__ activation_distance,
    const int     batch_size,
    const int     horizon,
    const int     n_spheres,
    const int     num_envs)
{
    __shared__ SQData sh[BLOCK];

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane    = tid & (WARP_SZ - 1);
    const int gid     = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int total   = batch_size * horizon * n_spheres;
    
    if (gid >= total) return;

    const int b_idx = gid / (horizon * n_spheres);
    const int h_idx = (gid - b_idx * (horizon * n_spheres)) / n_spheres;
    const int s_idx = gid - b_idx * horizon * n_spheres - h_idx * n_spheres;

    const float4 s_data = __ldg(reinterpret_cast<const float4*>(spheres) + gid);
    const float px = s_data.x, py = s_data.y, pz = s_data.z, pr = s_data.w;

    if (pr < 0.f) {
        out_dist[gid] = 0.f;
        return;
    }

    const int env_idx = env_query_idx != nullptr ? env_query_idx[b_idx] : 0;
    const int env_idx_clamped = clamp_env(env_idx, num_envs);
    
    const int sq_start = sq_env_starts[env_idx_clamped];
    const int sq_count = sq_env_counts[env_idx_clamped];

    float partial = 0.f;
    const float ad = activation_distance != nullptr
                     ? activation_distance[0] : 0.f;

    /* Warp-level iteration over all SQs in this environment */
    for (int base = 0; base < sq_count; base += BLOCK) {
        const int load_i = base + tid;
        if (load_i < sq_count)
            sh[tid] = all_sq_packed[sq_start + load_i];
        __syncthreads();

        const int tile_end = min(BLOCK, sq_count - base);
        for (int j = lane; j < tile_end; j += WARP_SZ) {
            if (sh[j]._pad == 0.f) continue;
            if (sq_aabb_miss(px, py, pz, pr, sh[j])) continue;
            
            const float sdf_curobo = -sq_sdf(px, py, pz, pr, sh[j]);
            if (sdf_curobo > 0.f) {
                if (ad > 0.f) {
                    if (sdf_curobo > ad) {
                        partial += sdf_curobo - 0.5f * ad;
                    } else {
                        partial += (0.5f / ad) * sdf_curobo * sdf_curobo;
                    }
                } else {
                    partial += sdf_curobo;
                }
            }
        }
        __syncthreads();
    }

    /* Warp reduction */
    partial += __shfl_down_sync(0xffffffff, partial, 16);
    partial += __shfl_down_sync(0xffffffff, partial,  8);
    partial += __shfl_down_sync(0xffffffff, partial,  4);
    partial += __shfl_down_sync(0xffffffff, partial,  2);
    partial += __shfl_down_sync(0xffffffff, partial,  1);

    if (lane == 0) {
        const float wt = weight != nullptr ? weight[0] : 1.f;
        out_dist[gid] = partial * wt;
    }
}

/* ── Batch environment swept-sphere kernel (handles all sweep steps internally) ─ */
__global__
void sphere_sq_batch_env_swept_kernel(
    const float*  __restrict__ sphere_pos,      // [batch, horizon, n_sph, 4]
    const float*  __restrict__ next_sphere_pos, // [batch, horizon, n_sph, 3] displacement
    float*        __restrict__ out_dist,        // [batch, horizon, n_spheres]
    const SQData* __restrict__ all_sq_packed,   // [num_envs * max_nobs]
    const int*    __restrict__ sq_env_starts,   // [num_envs]
    const int*    __restrict__ sq_env_counts,   // [num_envs]
    const int32_t*__restrict__ env_query_idx,   // [batch]
    const float*  __restrict__ weight,
    const float*  __restrict__ activation_distance,
    const float*  __restrict__ speed_dt,        // [batch, horizon]
    const int     batch_size,
    const int     horizon,
    const int     n_spheres,
    const int     num_envs,
    const int     sweep_steps,
    const bool    enable_speed_metric)
{
    __shared__ SQData sh[SQ_TILE];
    __shared__ float speed_scales[BLOCK];  // Cache speed scale per thread

    const int tid   = threadIdx.x;
    const int gid   = blockIdx.x * BLOCK + tid;
    const int total = batch_size * horizon * n_spheres;
    
    if (gid >= total) return;

    /* Decompose global index */
    const int b_idx = gid / (horizon * n_spheres);
    const int h_idx = (gid - b_idx * (horizon * n_spheres)) / n_spheres;
    const int s_idx = gid - b_idx * horizon * n_spheres - h_idx * n_spheres;

    /* Load current and next sphere positions */
    const float4 s_curr = __ldg(reinterpret_cast<const float4*>(sphere_pos) + gid);
    const float px0 = s_curr.x, py0 = s_curr.y, pz0 = s_curr.z, pr = s_curr.w;
    
    if (pr < 0.f) {
        out_dist[gid] = 0.f;
        return;
    }

    /* Load displacement to next waypoint */
    const int next_idx = (h_idx == horizon - 1) ? gid : gid + n_spheres;  // last step repeats
    const float4 s_next = __ldg(reinterpret_cast<const float4*>(next_sphere_pos) + next_idx);
    const float dx = s_next.x - px0, dy = s_next.y - py0, dz = s_next.z - pz0;
    
    /* Compute speed-scaling metric */
    float speed_scale = 1.f;
    if (enable_speed_metric) {
        const float step_len = sqrtf(dx * dx + dy * dy + dz * dz);
        const float dt = (speed_dt != nullptr && h_idx < horizon)
                        ? fmaxf(fabsf(speed_dt[b_idx * horizon + h_idx]), 1e-6f)
                        : 1.f;
        speed_scale = fminf(fmaxf(1.f + step_len / dt, 1.f), 50.f);
    }
    speed_scales[tid] = speed_scale;

    /* Determine environment */
    const int env_idx = env_query_idx != nullptr ? env_query_idx[b_idx] : 0;
    const int env_idx_clamped = clamp_env(env_idx, num_envs);
    
    const int sq_start = sq_env_starts[env_idx_clamped];
    const int sq_count = sq_env_counts[env_idx_clamped];

    float max_cost = 0.f;
    const float ad = activation_distance != nullptr
                     ? activation_distance[0] : 0.f;
    const float wt = weight != nullptr ? weight[0] : 1.f;
    const int eff_steps = max(sweep_steps, 1);

    /* Sample sphere along motion segment at (eff_steps+1) points */
    for (int step = 0; step <= eff_steps; ++step) {
        const float alpha = (float)step / (float)eff_steps;
        const float px = px0 + alpha * dx;
        const float py = py0 + alpha * dy;
        const float pz = pz0 + alpha * dz;

        float step_cost = 0.f;

        /* Tile loop over all SQs in this environment */
        for (int base = 0; base < sq_count; base += SQ_TILE) {
            if (tid < SQ_TILE) {
                const int load_i = sq_start + base + tid;
                if (load_i < sq_start + sq_count)
                    sh[tid] = all_sq_packed[load_i];
            }
            __syncthreads();

            const int tile_end = min(SQ_TILE, sq_count - base);
            for (int j = 0; j < tile_end; ++j) {
                if (sh[j]._pad == 0.f) continue;
                
                const float sdf_curobo = -sq_sdf(px, py, pz, pr, sh[j]);
                const float pos = fmaxf(sdf_curobo, 0.f);
                
                float cost;
                if (ad > 0.f) {
                    cost = (pos > ad) ? (pos - 0.5f * ad) : ((0.5f / ad) * pos * pos);
                } else {
                    cost = pos;
                }
                step_cost = fmaxf(step_cost, cost);
            }
            __syncthreads();
        }

        /* Apply speed metric and accumulate max */
        step_cost = step_cost * speed_scales[tid];
        max_cost = fmaxf(max_cost, step_cost);
    }

    out_dist[gid] = max_cost * wt;
}

/* ═════════════════════ Legacy ABI compatibility ═══════════════════════════
 *
 * geom_cuda.cpp still exports `superquadric_distance` and expects the legacy
 * launcher symbol below. Keep this symbol available and route the pairwise
 * distance path through the new sq_sdf implementation.
 *
 * Legacy Superquadric layout:
 *   struct { float3 center; float3 scale; float eps1; float eps2; }
 *
 * Output convention matches sq_sdf: positive outside, negative inside.
 * ═══════════════════════════════════════════════════════════════════════════ */

struct LegacySuperquadric {
    float3 center;
    float3 scale;
    float eps1;
    float eps2;
};

__global__
void sphere_sq_pairwise_distance_kernel(
    const float*  __restrict__ sphere_centers_f,  // reinterpreted float3→float*
    const float*  __restrict__ sphere_radii,
    const LegacySuperquadric* __restrict__ sqs,
    float*        __restrict__ distances,
    const int n)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    /* Coalesced scalar loads instead of uncoalesced float3 */
    const float cx = __ldg(&sphere_centers_f[gid * 3 + 0]);
    const float cy = __ldg(&sphere_centers_f[gid * 3 + 1]);
    const float cz = __ldg(&sphere_centers_f[gid * 3 + 2]);
    const float r  = fabsf(__ldg(&sphere_radii[gid]));

    const LegacySuperquadric q = sqs[gid];

    SQData sq;
    sq.cx = q.center.x;  sq.cy = q.center.y;  sq.cz = q.center.z;
    sq.sx = fmaxf(fabsf(q.scale.x), 1e-6f);
    sq.sy = fmaxf(fabsf(q.scale.y), 1e-6f);
    sq.sz = fmaxf(fabsf(q.scale.z), 1e-6f);
    sq.eps1 = fminf(fmaxf(fabsf(q.eps1), 0.05f), 4.0f);
    sq.eps2 = fminf(fmaxf(fabsf(q.eps2), 0.05f), 4.0f);

    distances[gid] = sq_sdf(cx, cy, cz, r, sq);
}

extern "C" void launch_sphere_sq_distance_kernel(
    float3*             sphere_centers,
    float*              sphere_radii,
    LegacySuperquadric* sqs,
    float*              distances,
    int  n,
    int  /*n_obs*/,   // retained for ABI compatibility; pairwise ignores it
    cudaStream_t stream)
{
    if (n <= 0) return;
    const int blocks = (n + BLOCK - 1) / BLOCK;
    sphere_sq_pairwise_distance_kernel<<<blocks, BLOCK, 0, stream>>>(
        reinterpret_cast<const float*>(sphere_centers),
        sphere_radii, sqs, distances, n);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Public entry points (registered with PyBind)
 * ═══════════════════════════════════════════════════════════════════════════ */

std::vector<torch::Tensor>
sphere_superquadric_clpt(
    const torch::Tensor sphere_position,
    torch::Tensor       distance,
    torch::Tensor       closest_point,
    torch::Tensor       sparsity_idx,
    const torch::Tensor weight,
    const torch::Tensor activation_distance,
    const torch::Tensor sq_params,
    const torch::Tensor sq_enable,
    const torch::Tensor n_env_sq,
    const torch::Tensor env_query_idx,
    const int  max_nobs,
    const int  batch_size,
    const int  horizon,
    const int  n_spheres,
    const bool compute_distance,
    const bool use_batch_env,
    const bool sum_collisions,
    const bool compute_esdf)
{
    (void)compute_distance;

    /* ── Type / contiguity normalisation ─────────────────────────────── */
    const auto sphere   = sphere_position.contiguous().to(torch::kFloat);
    const auto sq_p     = sq_params.contiguous().to(torch::kFloat);
    const auto sq_en    = sq_enable.contiguous().to(torch::kUInt8);
    const auto n_sq_i32 = n_env_sq.contiguous().to(torch::kInt32);

    const int   num_envs = std::max((int)sq_p.size(0), 1);
    const int64_t T      = (int64_t)batch_size * horizon * n_spheres;

    const auto fo  = torch::TensorOptions().dtype(torch::kFloat).device(sphere.device());
    const auto u8  = torch::TensorOptions().dtype(torch::kUInt8) .device(sphere.device());
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32) .device(sphere.device());
    const auto i64 = torch::TensorOptions().dtype(torch::kInt64) .device(sphere.device());

    auto dist_flat = compute_esdf ? torch::full({T}, -1e6f, fo)
                                  : torch::zeros({T}, fo);
    auto grad_flat  = torch::zeros({T, 4}, fo);
    auto spar_flat  = torch::zeros({T}, u8);
    const auto sph_flat = sphere.view({T, 4}).contiguous();

    /* Scalar weight and activation distance */
    const auto wt_s = weight.numel() > 0
                      ? weight.flatten().select(0, 0).to(fo)
                      : torch::ones({}, fo);
    const auto ad_s = activation_distance.numel() > 0
                      ? activation_distance.flatten().select(0, 0).to(fo)
                      : torch::zeros({}, fo);

    /* ── Per-query environment index (int32 for kernel) ──────────────── */
    auto env_q_i32 = use_batch_env
        ? env_query_idx.contiguous().to(sphere.device()).to(torch::kInt32).view({-1})
        : torch::zeros({batch_size}, i32);

    if (env_q_i32.numel() < batch_size) {
        auto pad = torch::zeros({batch_size}, i32);
        pad.slice(0, 0, env_q_i32.numel()).copy_(env_q_i32);
        env_q_i32 = pad;
    } else if (env_q_i32.numel() > batch_size) {
        env_q_i32 = env_q_i32.slice(0, 0, batch_size);
    }

    /* ── Enabled obstacle mask and pre-pack all environments ─────────── */
    const auto obs_range    = torch::arange(max_nobs, i32).view({1, max_nobs});
    const auto env_cnt_mask = (obs_range < n_sq_i32.view({-1, 1}));
    const auto en_mask      = sq_en.to(torch::kBool) & env_cnt_mask;

    /* Pre-compute SQ counts per environment */
    const auto sq_counts_per_env = torch::sum(en_mask.to(torch::kInt32), 1);

    /* Create packed buffer with all environments' SQs and offsets */
    std::vector<int> env_starts(num_envs);
    int offset = 0;
    for (int e = 0; e < num_envs; ++e) {
        env_starts[e] = offset;
        offset += max_nobs;  // Each environment gets max_nobs slots (disabled ones have _pad=0)
    }

    auto all_sq_packed = torch::empty({num_envs * max_nobs, 16}, fo);
    for (int e = 0; e < num_envs; ++e) {
        const auto sq_packed = pack_env_sq(sq_p, en_mask, e);
        all_sq_packed.slice(0, env_starts[e], env_starts[e] + max_nobs)
                     .copy_(sq_packed);
    }

    auto env_starts_tensor = torch::tensor(env_starts, 
                                            torch::TensorOptions().dtype(torch::kInt32).device(sphere.device()));
    auto sq_counts_tensor  = sq_counts_per_env.to(sphere.device());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    /* ── Single batch-environment kernel launch ──────────────────────── */
    const int total_threads = batch_size * horizon * n_spheres;
    int threadsPerBlock = total_threads;
    if (threadsPerBlock > 128) threadsPerBlock = 128;
    int blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    if (sum_collisions && !compute_esdf) {
        /* Sum-collisions path: single launch */
        sphere_sq_batch_env_sum_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            sph_flat.data_ptr<float>(),
            dist_flat.data_ptr<float>(),
            reinterpret_cast<const SQData*>(all_sq_packed.data_ptr<float>()),
            env_starts_tensor.data_ptr<int>(),
            sq_counts_tensor.data_ptr<int>(),
            env_q_i32.data_ptr<int32_t>(),
            wt_s.numel() > 0 ? wt_s.data_ptr<float>() : nullptr,
            ad_s.numel() > 0 ? ad_s.data_ptr<float>() : nullptr,
            batch_size, horizon, n_spheres, num_envs);
    } else {
        /* Min-distance path (max-cost and ESDF): single launch */
        if (compute_esdf) {
            sphere_sq_batch_env_min_kernel<true><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                sph_flat.data_ptr<float>(),
                dist_flat.data_ptr<float>(),
                reinterpret_cast<const SQData*>(all_sq_packed.data_ptr<float>()),
                env_starts_tensor.data_ptr<int>(),
                sq_counts_tensor.data_ptr<int>(),
                env_q_i32.data_ptr<int32_t>(),
                wt_s.numel() > 0 ? wt_s.data_ptr<float>() : nullptr,
                ad_s.numel() > 0 ? ad_s.data_ptr<float>() : nullptr,
                batch_size, horizon, n_spheres, num_envs);
        } else {
            sphere_sq_batch_env_min_kernel<false><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                sph_flat.data_ptr<float>(),
                dist_flat.data_ptr<float>(),
                reinterpret_cast<const SQData*>(all_sq_packed.data_ptr<float>()),
                env_starts_tensor.data_ptr<int>(),
                sq_counts_tensor.data_ptr<int>(),
                env_q_i32.data_ptr<int32_t>(),
                wt_s.numel() > 0 ? wt_s.data_ptr<float>() : nullptr,
                ad_s.numel() > 0 ? ad_s.data_ptr<float>() : nullptr,
                batch_size, horizon, n_spheres, num_envs);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    spar_flat = (dist_flat > 0.f).to(torch::kUInt8);

    distance.copy_(dist_flat.view({batch_size, horizon, n_spheres})
                            .to(distance.options()));
    closest_point.copy_(grad_flat.view({batch_size, horizon, n_spheres, 4})
                                 .to(closest_point.options()));
    sparsity_idx.copy_(spar_flat.view({batch_size, horizon, n_spheres})
                                .to(sparsity_idx.options()));

    return {distance, closest_point, sparsity_idx};
}

/* ─────────────────────────────────────────────────────────────────────────── */

std::vector<torch::Tensor>
swept_sphere_superquadric_clpt(
    const torch::Tensor sphere_position,
    torch::Tensor       distance,
    torch::Tensor       closest_point,
    torch::Tensor       sparsity_idx,
    const torch::Tensor weight,
    const torch::Tensor activation_distance,
    const torch::Tensor speed_dt,
    const torch::Tensor sq_params,
    const torch::Tensor sq_enable,
    const torch::Tensor n_env_sq,
    const torch::Tensor env_query_idx,
    const int  max_nobs,
    const int  batch_size,
    const int  horizon,
    const int  n_spheres,
    const int  sweep_steps,
    const bool enable_speed_metric,
    const bool compute_distance,
    const bool use_batch_env,
    const bool sum_collisions)
{
    (void)compute_distance;
    (void)sum_collisions;  // Swept always uses max aggregation

    /* ── Type / contiguity normalisation ─────────────────────────────── */
    const auto sphere   = sphere_position.contiguous().to(torch::kFloat);
    const auto sq_p     = sq_params.contiguous().to(torch::kFloat);
    const auto sq_en    = sq_enable.contiguous().to(torch::kUInt8);
    const auto n_sq_i32 = n_env_sq.contiguous().to(torch::kInt32);
    const auto speed    = speed_dt.numel() > 0 
                         ? speed_dt.contiguous().to(torch::kFloat).view({-1})
                         : torch::ones({batch_size * horizon}, torch::TensorOptions()
                             .dtype(torch::kFloat).device(sphere.device()));

    const int   num_envs = std::max((int)sq_p.size(0), 1);
    const int64_t T      = (int64_t)batch_size * horizon * n_spheres;

    const auto fo  = torch::TensorOptions().dtype(torch::kFloat).device(sphere.device());
    const auto u8  = torch::TensorOptions().dtype(torch::kUInt8) .device(sphere.device());
    const auto i32 = torch::TensorOptions().dtype(torch::kInt32) .device(sphere.device());

    auto dist_flat = torch::zeros({T}, fo);
    auto grad_flat = torch::zeros({T, 4}, fo);
    auto spar_flat = torch::zeros({T}, u8);
    const auto sph_flat = sphere.view({T, 4}).contiguous();

    /* Scalar weight and activation distance */
    const auto wt_s = weight.numel() > 0
                      ? weight.flatten().select(0, 0).to(fo)
                      : torch::ones({}, fo);
    const auto ad_s = activation_distance.numel() > 0
                      ? activation_distance.flatten().select(0, 0).to(fo)
                      : torch::zeros({}, fo);

    /* ── Per-query environment index (int32 for kernel) ──────────────── */
    auto env_q_i32 = use_batch_env
        ? env_query_idx.contiguous().to(sphere.device()).to(torch::kInt32).view({-1})
        : torch::zeros({batch_size}, i32);

    if (env_q_i32.numel() < batch_size) {
        auto pad = torch::zeros({batch_size}, i32);
        pad.slice(0, 0, env_q_i32.numel()).copy_(env_q_i32);
        env_q_i32 = pad;
    } else if (env_q_i32.numel() > batch_size) {
        env_q_i32 = env_q_i32.slice(0, 0, batch_size);
    }

    /* ── Enabled obstacle mask and pre-pack all environments ─────────── */
    const auto obs_range    = torch::arange(max_nobs, i32).view({1, max_nobs});
    const auto env_cnt_mask = (obs_range < n_sq_i32.view({-1, 1}));
    const auto en_mask      = sq_en.to(torch::kBool) & env_cnt_mask;

    /* Pre-compute SQ counts per environment */
    const auto sq_counts_per_env = torch::sum(en_mask.to(torch::kInt32), 1);

    /* Create packed buffer with all environments' SQs */
    std::vector<int> env_starts(num_envs);
    int offset = 0;
    for (int e = 0; e < num_envs; ++e) {
        env_starts[e] = offset;
        offset += max_nobs;
    }

    auto all_sq_packed = torch::empty({num_envs * max_nobs, 16}, fo);
    for (int e = 0; e < num_envs; ++e) {
        const auto sq_packed = pack_env_sq(sq_p, en_mask, e);
        all_sq_packed.slice(0, env_starts[e], env_starts[e] + max_nobs)
                     .copy_(sq_packed);
    }

    auto env_starts_tensor = torch::tensor(env_starts, 
                                            torch::TensorOptions().dtype(torch::kInt32).device(sphere.device()));
    auto sq_counts_tensor  = sq_counts_per_env.to(sphere.device());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    /* ── Compute next-position displacements ────────────────────────── */
    const auto sph_view  = sphere.view({batch_size, horizon, n_spheres, 4});
    const auto curr_pos  = sph_view.slice(3, 0, 3).contiguous();
    const auto curr_rad  = sph_view.select(3, 3).contiguous();
    
    /* Next positions: shift by 1 along horizon; last step repeats final */
    const auto next_pos = torch::cat({
        curr_pos.slice(1, 1, horizon),
        curr_pos.slice(1, horizon - 1, horizon)
    }, 1).contiguous();

    const auto disp = (next_pos - curr_pos).contiguous();

    /* ── Single batch-environment swept kernel launch ──────────────── */
    const int total_threads = batch_size * horizon * n_spheres;
    int threadsPerBlock = total_threads;
    if (threadsPerBlock > 128) threadsPerBlock = 128;
    int blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;

    sphere_sq_batch_env_swept_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        sph_flat.data_ptr<float>(),
        disp.view({-1, 3}).data_ptr<float>(),
        dist_flat.data_ptr<float>(),
        reinterpret_cast<const SQData*>(all_sq_packed.data_ptr<float>()),
        env_starts_tensor.data_ptr<int>(),
        sq_counts_tensor.data_ptr<int>(),
        env_q_i32.data_ptr<int32_t>(),
        wt_s.numel() > 0 ? wt_s.data_ptr<float>() : nullptr,
        ad_s.numel() > 0 ? ad_s.data_ptr<float>() : nullptr,
        speed.numel() > 0 ? speed.data_ptr<float>() : nullptr,
        batch_size, horizon, n_spheres, num_envs, sweep_steps, enable_speed_metric);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    /* ── Gradient computation (numerical FD for swept sphere) ──────── */
    if (sphere_position.requires_grad()) {
        constexpr float eps = 1e-3f;
        for (int ax = 0; ax < 3; ++ax) {
            auto qp = sph_flat.clone(); 
            qp.slice(1, ax * total_threads / 4, (ax + 1) * total_threads / 4).add_(eps);
            auto qm = sph_flat.clone(); 
            qm.slice(1, ax * total_threads / 4, (ax + 1) * total_threads / 4).sub_(eps);

            auto dist_p = torch::zeros({T}, fo);
            auto dist_m = torch::zeros({T}, fo);

            /* Launch kernels for +eps and -eps perturbations */
            sphere_sq_batch_env_swept_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                qp.data_ptr<float>(),
                disp.view({-1, 3}).data_ptr<float>(),
                dist_p.data_ptr<float>(),
                reinterpret_cast<const SQData*>(all_sq_packed.data_ptr<float>()),
                env_starts_tensor.data_ptr<int>(),
                sq_counts_tensor.data_ptr<int>(),
                env_q_i32.data_ptr<int32_t>(),
                wt_s.numel() > 0 ? wt_s.data_ptr<float>() : nullptr,
                ad_s.numel() > 0 ? ad_s.data_ptr<float>() : nullptr,
                speed.numel() > 0 ? speed.data_ptr<float>() : nullptr,
                batch_size, horizon, n_spheres, num_envs, sweep_steps, enable_speed_metric);

            sphere_sq_batch_env_swept_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                qm.data_ptr<float>(),
                disp.view({-1, 3}).data_ptr<float>(),
                dist_m.data_ptr<float>(),
                reinterpret_cast<const SQData*>(all_sq_packed.data_ptr<float>()),
                env_starts_tensor.data_ptr<int>(),
                sq_counts_tensor.data_ptr<int>(),
                env_q_i32.data_ptr<int32_t>(),
                wt_s.numel() > 0 ? wt_s.data_ptr<float>() : nullptr,
                ad_s.numel() > 0 ? ad_s.data_ptr<float>() : nullptr,
                speed.numel() > 0 ? speed.data_ptr<float>() : nullptr,
                batch_size, horizon, n_spheres, num_envs, sweep_steps, enable_speed_metric);

            grad_flat.select(1, ax).copy_((dist_p - dist_m) * (0.5f / eps));
        }
        grad_flat.select(1, 3).zero_();
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    spar_flat = (dist_flat > 0.f).to(torch::kUInt8);

    distance.copy_(dist_flat.view({batch_size, horizon, n_spheres})
                            .to(distance.options()));
    closest_point.copy_(grad_flat.view({batch_size, horizon, n_spheres, 4})
                                 .to(closest_point.options()));
    sparsity_idx.copy_(spar_flat.view({batch_size, horizon, n_spheres})
                                .to(sparsity_idx.options()));

    return {distance, closest_point, sparsity_idx};
}