/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/*
 * NOTE: This file is NOT compiled and is kept for historical reference only.
 *
 * The active superquadric collision implementation is in:
 *   superquadric_radial_distance_kernel.cu
 *
 * That file implements analytical SDF + gradient using Newton radial projection
 * and does not depend on any external library.
 */

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if __has_include(<openGJK_GPU.h>)
#include <openGJK_GPU.h>
#define CUROBO_HAS_OPENGJK 1
#else
#define CUROBO_HAS_OPENGJK 0
#endif

#if CUROBO_HAS_OPENGJK

namespace
{
inline float collision_cost_from_sdf(const float sdf, const float activation_distance)
{
  if (sdf <= 0.0f)
  {
    return 0.0f;
  }

  if (activation_distance > 0.0f)
  {
    if (sdf > activation_distance)
    {
      return sdf - (0.5f * activation_distance);
    }
    return (0.5f / activation_distance) * sdf * sdf;
  }

  return sdf;
}

inline int clamp_env_idx(const int env_idx, const int max_envs)
{
  if (max_envs <= 0)
  {
    return 0;
  }
  if (env_idx < 0)
  {
    return 0;
  }
  if (env_idx >= max_envs)
  {
    return max_envs - 1;
  }
  return env_idx;
}

torch::Tensor collision_cost_from_sdf_tensor(
  const torch::Tensor& sdf,
  const torch::Tensor& activation_distance)
{
  auto positive = torch::relu(sdf);
  auto activation_safe = torch::clamp(activation_distance, 1.0e-12);
  auto linear = positive - (0.5 * activation_safe);
  auto quadratic = positive * positive * (0.5 / activation_safe);
  auto smooth = torch::where(positive > activation_safe, linear, quadratic);
  auto has_activation = (activation_distance > 0.0).to(positive.scalar_type());
  return has_activation * smooth + (1.0 - has_activation) * positive;
}

torch::Tensor evaluate_all_query_superquadric_values(
  const torch::Tensor& query_spheres,     // [Q,4], float32 cuda
  const torch::Tensor& sq_params,         // [n_env,max_nobs,12], float32 cuda
  const torch::Tensor& enabled_mask,      // [n_env,max_nobs], bool cuda
  const torch::Tensor& query_env_idx,     // [Q], int64 cuda
  const int env_idx,
  const int max_nobs,
  const torch::Tensor& weight,
  const torch::Tensor& activation_distance,
  const bool sum_collisions,
  const bool compute_esdf,
  cudaStream_t stream)
{
  const int64_t q_count = query_spheres.size(0);
  auto options = query_spheres.options().dtype(torch::kFloat);
  if (q_count == 0)
  {
    return torch::empty({0}, options);
  }

  const int clamped_env_idx = clamp_env_idx(env_idx, static_cast<int>(sq_params.size(0)));
  auto env_query_mask = (query_env_idx == static_cast<int64_t>(clamped_env_idx));
  auto env_query_mask_f = env_query_mask.to(options.dtype());

  auto values = compute_esdf ? torch::full({q_count}, -1.0e6f, options)
                             : torch::zeros({q_count}, options);
  auto raw_signed_distance = torch::empty({q_count}, options);

  for (int obs_idx = 0; obs_idx < max_nobs; ++obs_idx)
  {
    auto sq_param = sq_params.select(0, clamped_env_idx).select(0, obs_idx);
    compute_superquadric_sphere_signed_distance_batch_single_sq(
      sq_param.data_ptr<float>(),
      query_spheres.data_ptr<float>(),
      static_cast<int>(q_count),
      raw_signed_distance.data_ptr<float>(),
      stream);

    // Conservative geometric lower bound on SQ-vs-sphere signed distance:
    // d >= ||c_sphere - c_sq|| - (R_sq_outer + r_sphere), where
    // R_sq_outer = max(a1,a2,a3). This bound is cheap and prevents numeric
    // pathologies in GJK/EPA from reporting far-field penetrations.
    // sq_param layout: [sx,sy,sz, eps1,eps2, cx,cy,cz, qx,qy,qz,qw]
    auto sq_center = sq_param.slice(0, 5, 8).view({1, 3});
    auto sq_outer_radius = torch::max(sq_param.slice(0, 0, 3)).view({1});
    auto sphere_centers = query_spheres.slice(1, 0, 3);
    auto sphere_radii = torch::abs(query_spheres.select(1, 3));
    auto center_delta = sphere_centers - sq_center;
    auto center_distance = torch::sqrt(torch::sum(center_delta * center_delta, 1));
    auto lower_bound = center_distance - (sq_outer_radius + sphere_radii);
    raw_signed_distance = torch::maximum(raw_signed_distance, lower_bound);

    // Tighter local-AABB lower bound: rotate delta to SQ local frame, then
    // compute box distance. The SQ surface is contained within its AABB for
    // ε≤2, so any sphere outside the AABB is definitively outside the SQ.
    // This overrides spurious GJK penetrations for boxy (small ε) shapes.
    {
      // Normalise quaternion [qx,qy,qz,qw] stored at indices 8-11.
      auto sq_q = sq_param.slice(0, 8, 12);
      auto qn = sq_q / torch::sqrt((sq_q * sq_q).sum()).clamp_min(1e-8f);
      // 0-dim tensors for each quaternion component.
      auto qx_t = qn[0], qy_t = qn[1], qz_t = qn[2], qw_t = qn[3];

      // Standard rotation matrix R (local→world) from unit quaternion:
      //   R[:,0] = {1-2(y²+z²), 2(xy+wz), 2(xz-wy)}
      //   R[:,1] = {2(xy-wz), 1-2(x²+z²), 2(yz+wx)}
      //   R[:,2] = {2(xz+wy), 2(yz-wx), 1-2(x²+y²)}
      // p_local = R^T * p_world, so (R^T*v)_i = R_col_i · v.
      auto r00 = 1.f - 2.f*(qy_t*qy_t + qz_t*qz_t);
      auto r10 = 2.f*(qx_t*qy_t + qw_t*qz_t);
      auto r20 = 2.f*(qx_t*qz_t - qw_t*qy_t);
      auto r01 = 2.f*(qx_t*qy_t - qw_t*qz_t);
      auto r11 = 1.f - 2.f*(qx_t*qx_t + qz_t*qz_t);
      auto r21 = 2.f*(qy_t*qz_t + qw_t*qx_t);
      auto r02 = 2.f*(qx_t*qz_t + qw_t*qy_t);
      auto r12 = 2.f*(qy_t*qz_t - qw_t*qx_t);
      auto r22 = 1.f - 2.f*(qx_t*qx_t + qy_t*qy_t);

      // center_delta is already computed as sphere_centers - sq_center [Q,3].
      auto cdx = center_delta.select(1, 0);
      auto cdy = center_delta.select(1, 1);
      auto cdz = center_delta.select(1, 2);

      // Apply R^T to each row of center_delta → local-frame coordinates.
      auto lx = r00*cdx + r10*cdy + r20*cdz;
      auto ly = r01*cdx + r11*cdy + r21*cdz;
      auto lz = r02*cdx + r12*cdy + r22*cdz;

      // AABB half-extents in local frame.
      auto sx_t = sq_param[0], sy_t = sq_param[1], sz_t = sq_param[2];

      // Per-axis excess beyond the box half-extent (clamped to ≥0).
      auto bx = torch::clamp(lx.abs() - sx_t, 0.f);
      auto by = torch::clamp(ly.abs() - sy_t, 0.f);
      auto bz = torch::clamp(lz.abs() - sz_t, 0.f);

      // AABB surface distance minus sphere radius → signed lower bound.
      // Positive means the sphere is definitively outside the SQ.
      auto lb_box = torch::sqrt(bx*bx + by*by + bz*bz) - sphere_radii;
      raw_signed_distance = torch::maximum(raw_signed_distance, lb_box);
    }

    // OpenGJK returns positive outside / negative penetration. CuRobo uses
    // positive inside / negative outside, so we negate here.
    auto sdf = -raw_signed_distance;
    auto obs_enabled = enabled_mask.select(0, clamped_env_idx).select(0, obs_idx).to(options.dtype());
    auto active_mask_f = env_query_mask_f * obs_enabled;
    auto active_mask = active_mask_f > 0.5;

    if (compute_esdf)
    {
      values = torch::where(active_mask, torch::maximum(values, sdf), values);
    }
    else
    {
      auto cost = collision_cost_from_sdf_tensor(sdf, activation_distance);
      if (sum_collisions)
      {
        values = values + (cost * active_mask_f);
      }
      else
      {
        values = torch::where(active_mask, torch::maximum(values, cost), values);
      }
    }
  }

  if (!compute_esdf)
  {
    values = values * weight;
  }

  return values;
}
} // namespace

std::vector<torch::Tensor>
sphere_superquadric_clpt(const torch::Tensor sphere_position, // [b,h,n,4]
                         torch::Tensor distance,
                         torch::Tensor closest_point,         // gradient buffer [b,h,n,4]
                         torch::Tensor sparsity_idx,
                         const torch::Tensor weight,
                         const torch::Tensor activation_distance,
                         const torch::Tensor sq_params,       // [n_env,max_nobs,12]
                         const torch::Tensor sq_enable,       // [n_env,max_nobs]
                         const torch::Tensor n_env_sq,        // [n_env]
                         const torch::Tensor env_query_idx,   // [b]
                         const int max_nobs,
                         const int batch_size,
                         const int horizon,
                         const int n_spheres,
                         const bool compute_distance,
                         const bool use_batch_env,
                         const bool sum_collisions,
                         const bool compute_esdf)
{
  (void)compute_distance;

  auto sphere = sphere_position.contiguous().to(torch::kFloat);
  auto sq_params_f = sq_params.contiguous().to(torch::kFloat);
  auto sq_enable_u8 = sq_enable.contiguous().to(torch::kUInt8);
  auto n_env_sq_i32 = n_env_sq.contiguous().to(torch::kInt32);

  const int num_envs = std::max(static_cast<int>(sq_params_f.size(0)), 1);

  const auto float_options = torch::TensorOptions().dtype(torch::kFloat).device(sphere.device());
  const auto uint8_options = torch::TensorOptions().dtype(torch::kUInt8).device(sphere.device());
  const auto i32_options = torch::TensorOptions().dtype(torch::kInt32).device(sphere.device());
  const auto i64_options = torch::TensorOptions().dtype(torch::kInt64).device(sphere.device());
  const int64_t total_queries = static_cast<int64_t>(batch_size) * horizon * n_spheres;
  auto distance_flat = compute_esdf ? torch::full({total_queries}, -1.0e6f, float_options)
                                    : torch::zeros({total_queries}, float_options);
  auto grad_flat = torch::zeros({total_queries, 4}, float_options);
  auto sparsity_flat = torch::zeros({total_queries}, uint8_options);
  auto sphere_flat = sphere.view({total_queries, 4}).contiguous();
  auto weight_scalar = weight.numel() > 0 ? weight.flatten().select(0, 0).to(float_options)
                                          : torch::ones({}, float_options);
  auto activation_scalar =
    activation_distance.numel() > 0
      ? activation_distance.flatten().select(0, 0).to(float_options)
      : torch::zeros({}, float_options);

  auto env_query_idx_i64 = use_batch_env
                             ? env_query_idx.contiguous().to(sphere.device()).to(torch::kInt64).view({-1})
                             : torch::zeros({batch_size}, i64_options);
  if (env_query_idx_i64.numel() == 0)
  {
    env_query_idx_i64 = torch::zeros({batch_size}, i64_options);
  }
  else if (env_query_idx_i64.numel() < batch_size)
  {
    auto padded = torch::zeros({batch_size}, i64_options);
    padded.slice(0, 0, env_query_idx_i64.numel()).copy_(env_query_idx_i64);
    env_query_idx_i64 = padded;
  }
  else if (env_query_idx_i64.numel() > batch_size)
  {
    env_query_idx_i64 = env_query_idx_i64.slice(0, 0, batch_size);
  }

  auto query_ids = torch::arange(total_queries, i64_options);
  auto batch_ids = torch::floor_divide(query_ids, static_cast<int64_t>(horizon) * n_spheres);
  auto query_env_idx = env_query_idx_i64.index_select(0, batch_ids);
  query_env_idx = torch::clamp(query_env_idx, static_cast<int64_t>(0), static_cast<int64_t>(num_envs - 1));

  auto obs_range = torch::arange(max_nobs, i32_options).view({1, max_nobs});
  auto env_count_mask = obs_range < n_env_sq_i32.view({-1, 1});
  auto enabled_mask = sq_enable_u8.to(torch::kBool) & env_count_mask;

  const float grad_eps = 1.0e-3f;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (int env_idx = 0; env_idx < num_envs; ++env_idx)
  {
    auto env_values = evaluate_all_query_superquadric_values(
      sphere_flat,
      sq_params_f,
      enabled_mask,
      query_env_idx,
      env_idx,
      max_nobs,
      weight_scalar,
      activation_scalar,
      sum_collisions,
      compute_esdf,
      stream);

    if (compute_esdf)
    {
      distance_flat = torch::where(
        query_env_idx == static_cast<int64_t>(env_idx),
        env_values,
        distance_flat);
    }
    else
    {
      distance_flat = distance_flat + env_values;
    }

    if (sphere_position.requires_grad())
    {
      auto env_grad = torch::zeros({sphere_flat.size(0), 4}, float_options);
      for (int axis = 0; axis < 3; ++axis)
      {
        auto q_plus = sphere_flat.clone();
        auto q_minus = sphere_flat.clone();
        q_plus.select(1, axis).add_(grad_eps);
        q_minus.select(1, axis).sub_(grad_eps);

        auto v_plus = evaluate_all_query_superquadric_values(
          q_plus,
          sq_params_f,
          enabled_mask,
          query_env_idx,
          env_idx,
          max_nobs,
          weight_scalar,
          activation_scalar,
          sum_collisions,
          compute_esdf,
          stream);
        auto v_minus = evaluate_all_query_superquadric_values(
          q_minus,
          sq_params_f,
          enabled_mask,
          query_env_idx,
          env_idx,
          max_nobs,
          weight_scalar,
          activation_scalar,
          sum_collisions,
          compute_esdf,
          stream);
        env_grad.select(1, axis).copy_((v_plus - v_minus) * (0.5f / grad_eps));
      }
      env_grad.select(1, 3).zero_();
      auto env_mask = (query_env_idx == static_cast<int64_t>(env_idx)).to(float_options).unsqueeze(1);
      grad_flat = grad_flat + env_grad * env_mask;
    }
  }

  sparsity_flat = (distance_flat > 0.0).to(torch::kUInt8);

  distance.copy_(distance_flat.view({batch_size, horizon, n_spheres}).to(distance.options()));
  closest_point.copy_(
    grad_flat.view({batch_size, horizon, n_spheres, 4}).to(closest_point.options()));
  sparsity_idx.copy_(
    sparsity_flat.view({batch_size, horizon, n_spheres}).to(sparsity_idx.options()));

  return {distance, closest_point, sparsity_idx};
}

#else

std::vector<torch::Tensor>
sphere_superquadric_clpt(const torch::Tensor sphere_position, // [b,h,n,4]
                         torch::Tensor distance,
                         torch::Tensor closest_point,
                         torch::Tensor sparsity_idx,
                         const torch::Tensor weight,
                         const torch::Tensor activation_distance,
                         const torch::Tensor sq_params,
                         const torch::Tensor sq_enable,
                         const torch::Tensor n_env_sq,
                         const torch::Tensor env_query_idx,
                         const int max_nobs,
                         const int batch_size,
                         const int horizon,
                         const int n_spheres,
                         const bool compute_distance,
                         const bool use_batch_env,
                         const bool sum_collisions,
                         const bool compute_esdf)
{
  (void)sphere_position;
  (void)distance;
  (void)closest_point;
  (void)sparsity_idx;
  (void)weight;
  (void)activation_distance;
  (void)sq_params;
  (void)sq_enable;
  (void)n_env_sq;
  (void)env_query_idx;
  (void)max_nobs;
  (void)batch_size;
  (void)horizon;
  (void)n_spheres;
  (void)compute_distance;
  (void)use_batch_env;
  (void)sum_collisions;
  (void)compute_esdf;

  TORCH_CHECK(
    false,
    "Superquadric support is unavailable because openGJK headers were not found at build time.");
}

#endif
