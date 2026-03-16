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
};
static_assert(sizeof(SQData) == 48, "SQData must be 48 bytes");

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


/* ═══════════════════════════ Core SDF ══════════════════════════════════════
 *
 * Returns the first-order signed-distance approximation for a sphere against
 * a superquadric, clamped by the conservative geometric lower bound.
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
 *   SDF ≈ (F − 1) / ‖∇F‖  − r_sphere
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
    // For eps ≤ 2 the SQ surface is always inside its axis-aligned bounding box,
    // so d_AABB(p) ≤ d_SQ(p) — a valid conservative lower bound.
    const float dx_box = fmaxf(fabsf(lx) - sq.sx, 0.f);
    const float dy_box = fmaxf(fabsf(ly) - sq.sy, 0.f);
    const float dz_box = fmaxf(fabsf(lz) - sq.sz, 0.f);
    const float lb_box = sqrtf(fmaf(dx_box, dx_box, fmaf(dy_box, dy_box, dz_box * dz_box))) - pr;

    return fmaxf(sdf_approx, fmaxf(lb, lb_box));
}

/* ═══════════════════════════ CUDA kernels ══════════════════════════════════ */

static constexpr int BLOCK   = 128;   // threads per block
static constexpr int SQ_TILE = 32;    // SQs loaded into shared memory per tile
                                       // must satisfy SQ_TILE ≤ BLOCK
static constexpr float MIN_RADIUS = 1e-2f;  // Avoid degenerate SQ axes

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
            /* Unroll hint; compiler may partially unroll the inner loop */
            #pragma unroll 4
            for (int j = 0; j < tile_end; ++j)
                min_d = fminf(min_d, sq_sdf(px, py, pz, pr, sh[j]));
        }
        __syncthreads();
    }

    if (valid)
        out_dist[gid] = min_d;
}

/* ── Kernel B: sum of smoothed collision costs over all obstacles ────────────
 *
 * Folds the collision_cost_from_sdf function into the kernel to avoid an
 * extra torch kernel launch in the sum-collisions path.
 * Weight is applied in the C++ wrapper to keep the kernel generic.
 */
__global__
void sphere_sq_sum_cost_kernel(
    const float*  __restrict__ spheres,   // [n_spheres, 4]
    const SQData* __restrict__ sq_arr,    // [n_obs]
    float*        __restrict__ out_cost,  // [n_spheres]  (unweighted)
    const int     n_spheres,
    const int     n_obs,
    const float   act_dist)               // activation distance scalar (≥ 0)
{
    __shared__ SQData sh[SQ_TILE];

    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * BLOCK + tid;
    const bool valid = (gid < n_spheres);

    float px = 0.f, py = 0.f, pz = 0.f, pr = 0.f;
    if (valid) {
        const float4 s = __ldg(reinterpret_cast<const float4*>(spheres) + gid);
        if (s.w < 0.f) { out_cost[gid] = 0.f; return; }    // disabled sphere
        px = s.x; py = s.y; pz = s.z; pr = s.w;
    }

    float total = 0.f;

    for (int base = 0; base < n_obs; base += SQ_TILE) {
        if (tid < SQ_TILE) {
            const int load_i = base + tid;
            if (load_i < n_obs)
                sh[tid] = sq_arr[load_i];
        }
        __syncthreads();

        if (valid) {
            const int tile_end = min(SQ_TILE, n_obs - base);
            #pragma unroll 4
            for (int j = 0; j < tile_end; ++j) {
                /* Negate: positive = penetrating (CuRobo convention) */
                const float sdf = -sq_sdf(px, py, pz, pr, sh[j]);
                if (sdf > 0.f) {
                    /* Smooth quadratic–linear collision cost */
                    total += (act_dist > 0.f)
                        ? ((sdf > act_dist)
                            ? sdf - 0.5f * act_dist
                            : (0.5f / act_dist) * sdf * sdf)
                        : sdf;
                }
            }
        }
        __syncthreads();
    }

    if (valid)
        out_cost[gid] = total;
}

/* ═══════════════════════════ C++ helpers ═══════════════════════════════════ */

namespace {

static int clamp_env(int e, int maxe)
{
    return std::max(0, std::min(e, maxe - 1));
}

/* ── pack_env_sq ─────────────────────────────────────────────────────────────
 *
 * Select enabled obstacles for one environment and repack them into
 * SQData layout [cx,cy,cz,sx,sy,sz,eps1,eps2] (from input layout
 * [sx,sy,sz,eps1,eps2,cx,cy,cz]).  Clamping is applied here once, not
 * inside every kernel invocation.
 *
 * Returns a contiguous float32 GPU tensor of shape [n_obs, 8].
 * Returns an empty tensor if no obstacles are enabled.
 */
torch::Tensor pack_env_sq(
    const torch::Tensor& sq_params,
    const torch::Tensor& enabled_mask,
    const int env_idx)
{
    auto dev_opts = torch::TensorOptions()
                        .dtype(torch::kFloat)
                        .device(sq_params.device());

    const auto raw  = sq_params[env_idx];
    const auto mask = enabled_mask[env_idx];

    const auto idx_2d = mask.nonzero();
    if (idx_2d.size(0) == 0)
        return torch::empty({0, 12}, dev_opts);  // Bug 3 fix: was {0,8}

    const auto idx = idx_2d.squeeze(1);
    const auto sel = raw.index_select(0, idx)
                        .to(torch::kFloat)
                        .contiguous();

    // Input layout: [sx,sy,sz, eps1,eps2, cx,cy,cz, qx,qy,qz,qw]
    // (Python stores quaternion as [qx,qy,qz,qw]; SQData expects [qw,qx,qy,qz])
    const int64_t n_obs = idx.size(0);
    auto out = torch::empty({n_obs, 12}, dev_opts);

    out.slice(1, 0, 3).copy_(sel.slice(1, 5, 8));   // cx,cy,cz
    out.slice(1, 3, 6).copy_(
        torch::clamp(torch::abs(sel.slice(1, 0, 3)), MIN_RADIUS));  // sx,sy,sz
    out.select(1, 6).copy_(
        torch::clamp(torch::abs(sel.select(1, 3)), 0.05f, 4.0f));   // eps1
    out.select(1, 7).copy_(
        torch::clamp(torch::abs(sel.select(1, 4)), 0.05f, 4.0f));   // eps2

    // Quaternion — normalise and reorder [qx,qy,qz,qw] → [qw,qx,qy,qz] for SQData
    auto q      = sel.slice(1, 8, 12).to(torch::kFloat);
    auto q_norm = torch::clamp(torch::norm(q, 2, 1, true), 1e-6f);
    auto q_n    = (q / q_norm).contiguous();
    out.select(1, 8).copy_(q_n.select(1, 3));    // qw  (was at index 3)
    out.slice(1, 9, 12).copy_(q_n.slice(1, 0, 3));  // qx,qy,qz (were at 0,1,2)

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

    /* ── Pack enabled SQ descriptors for this environment ────────────── */
    const auto sq_packed = pack_env_sq(sq_params, enabled_mask, cenv);
    const int n_obs = (int)sq_packed.size(0);
    if (n_obs == 0)
        return values;

    auto raw = torch::empty({q}, opt);    // kernel output buffer

    const int blocks   = ((int)q + BLOCK - 1) / BLOCK;
    const auto* sq_ptr = reinterpret_cast<const SQData*>(sq_packed.data_ptr<float>());
    const float* sp    = query_spheres.data_ptr<float>();

    /* ── Extract scalar parameters ──────────────────────────────────── */
    const float ad = act_dist.numel() > 0
                     ? act_dist.flatten().select(0, 0).item<float>() : 0.f;
    const float wt = weight.numel() > 0
                     ? weight.flatten().select(0, 0).item<float>()   : 1.f;

    /* ════ Sum-collisions path ════════════════════════════════════════ */
    if (sum_collisions && !compute_esdf) {
        sphere_sq_sum_cost_kernel<<<blocks, BLOCK, 0, stream>>>(
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

} // anonymous namespace

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

    /* ── Per-query environment index ─────────────────────────────────── */
    auto env_q = use_batch_env
        ? env_query_idx.contiguous().to(sphere.device()).to(torch::kInt64).view({-1})
        : torch::zeros({batch_size}, i64);

    if (env_q.numel() == 0) {
        env_q = torch::zeros({batch_size}, i64);
    } else if (env_q.numel() < batch_size) {
        auto pad = torch::zeros({batch_size}, i64);
        pad.slice(0, 0, env_q.numel()).copy_(env_q);
        env_q = pad;
    } else if (env_q.numel() > batch_size) {
        env_q = env_q.slice(0, 0, batch_size);
    }

    /* Broadcast batch-level env index to all (batch*horizon*n_spheres) queries */
    const auto qids  = torch::arange(T, i64);
    const auto bids  = torch::floor_divide(qids, (int64_t)horizon * n_spheres);
    auto q_env = env_q.index_select(0, bids);
    q_env = torch::clamp(q_env, (int64_t)0, (int64_t)(num_envs - 1));

    /* ── Enabled obstacle mask [nenv, max_nobs] ──────────────────────── */
    const auto obs_range    = torch::arange(max_nobs, i32).view({1, max_nobs});
    const auto env_cnt_mask = (obs_range < n_sq_i32.view({-1, 1}));
    const auto en_mask      = sq_en.to(torch::kBool) & env_cnt_mask;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    /* ── Per-environment evaluation ──────────────────────────────────── */
    for (int e = 0; e < num_envs; ++e) {
        const auto ev = evaluate_all_sq(
            sph_flat, sq_p, en_mask, q_env, e,
            wt_s, ad_s, sum_collisions, compute_esdf, stream);

        if (compute_esdf)
            dist_flat = torch::where(q_env == (int64_t)e, ev, dist_flat);
        else
            dist_flat = dist_flat + ev;

        /* ── Numerical gradient (central finite differences) ──────────
         *
         * NOTE: 6 extra kernel launches per gradient call.  Analytical
         * gradients of the SDF approximation are tractable and would be
         * significantly faster — left as a future improvement.
         */
        if (sphere_position.requires_grad()) {
            constexpr float eps = 1e-3f;
            auto eg = torch::zeros({T, 4}, fo);

            for (int ax = 0; ax < 3; ++ax) {
                auto qp = sph_flat.clone(); qp.select(1, ax).add_(eps);
                auto qm = sph_flat.clone(); qm.select(1, ax).sub_(eps);

                const auto vp = evaluate_all_sq(
                    qp, sq_p, en_mask, q_env, e,
                    wt_s, ad_s, sum_collisions, compute_esdf, stream);
                const auto vm = evaluate_all_sq(
                    qm, sq_p, en_mask, q_env, e,
                    wt_s, ad_s, sum_collisions, compute_esdf, stream);

                eg.select(1, ax).copy_((vp - vm) * (0.5f / eps));
            }
            eg.select(1, 3).zero_();

            const auto emask_f = (q_env == (int64_t)e).to(fo).unsqueeze(1);
            grad_flat = grad_flat + eg * emask_f;
        }
    }

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

    auto dist_flat = torch::zeros({T}, fo);
    auto grad_flat = torch::zeros({T, 4}, fo);
    auto spar_flat = torch::zeros({T}, u8);

    auto env_q = use_batch_env
        ? env_query_idx.contiguous().to(sphere.device()).to(torch::kInt64).view({-1})
        : torch::zeros({batch_size}, i64);

    if (env_q.numel() == 0) {
        env_q = torch::zeros({batch_size}, i64);
    } else if (env_q.numel() < batch_size) {
        auto pad = torch::zeros({batch_size}, i64);
        pad.slice(0, 0, env_q.numel()).copy_(env_q);
        env_q = pad;
    } else if (env_q.numel() > batch_size) {
        env_q = env_q.slice(0, 0, batch_size);
    }

    const auto qids = torch::arange(T, i64);
    const auto bids = torch::floor_divide(qids, (int64_t)horizon * n_spheres);
    auto q_env      = env_q.index_select(0, bids);
    q_env = torch::clamp(q_env, (int64_t)0, (int64_t)(num_envs - 1));

    const auto obs_range    = torch::arange(max_nobs, i32).view({1, max_nobs});
    const auto env_cnt_mask = (obs_range < n_sq_i32.view({-1, 1}));
    const auto en_mask      = sq_en.to(torch::kBool) & env_cnt_mask;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dist_flat = evaluate_swept_sq(
        sphere, sq_p, en_mask, q_env,
        weight, activation_distance, speed_dt,
        sweep_steps, enable_speed_metric,
        sum_collisions, false, stream);

    if (sphere_position.requires_grad()) {
        constexpr float eps = 1e-3f;
        for (int ax = 0; ax < 3; ++ax) {
            auto qp = sphere.clone(); qp.select(3, ax).add_(eps);
            auto qm = sphere.clone(); qm.select(3, ax).sub_(eps);

            const auto vp = evaluate_swept_sq(
                qp, sq_p, en_mask, q_env,
                weight, activation_distance, speed_dt,
                sweep_steps, enable_speed_metric,
                sum_collisions, false, stream);
            const auto vm = evaluate_swept_sq(
                qm, sq_p, en_mask, q_env,
                weight, activation_distance, speed_dt,
                sweep_steps, enable_speed_metric,
                sum_collisions, false, stream);

            grad_flat.select(1, ax).copy_((vp - vm) * (0.5f / eps));
        }
        grad_flat.select(1, 3).zero_();
    }

    spar_flat = (dist_flat > 0.f).to(torch::kUInt8);

    distance.copy_(dist_flat.view({batch_size, horizon, n_spheres})
                            .to(distance.options()));
    closest_point.copy_(grad_flat.view({batch_size, horizon, n_spheres, 4})
                                 .to(closest_point.options()));
    sparsity_idx.copy_(spar_flat.view({batch_size, horizon, n_spheres})
                                .to(sparsity_idx.options()));

    return {distance, closest_point, sparsity_idx};
}