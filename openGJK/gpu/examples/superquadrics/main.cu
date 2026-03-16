/**
 * @file main.cu
 * @brief Superquadric-sphere signed distance example and sanity check.
 *
 * This example uses the GPU API:
 * compute_superquadric_sphere_signed_distance(...)
 */

#include "openGJK_GPU.h"

#include <cmath>
#include <cstdio>

static float float3_length(float3 value) {
	return sqrtf(value.x * value.x + value.y * value.y + value.z * value.z);
}

static float3 normalize_float3(float3 value) {
	const float length = float3_length(value);
	if (length < 1.0e-8f) {
		return make_float3(1.0f, 0.0f, 0.0f);
	}
	return make_float3(value.x / length, value.y / length, value.z / length);
}

static float4 make_identity_quaternion() {
	return make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

static float4 make_z_rotation_quaternion(float angle_rad) {
	const float half_angle = 0.5f * angle_rad;
	return make_float4(0.0f, 0.0f, sinf(half_angle), cosf(half_angle));
}

static void init_identity_superquadric_as_sphere(
		Superquadric* sq,
		float radius,
		float3 center) {
	sq->radii = make_float3(radius, radius, radius);
	sq->eps = make_float2(1.0f, 1.0f);
	sq->pos = center;
	sq->quat = make_identity_quaternion();
}

static void init_superquadric(
		Superquadric* sq,
		float3 radii,
		float2 eps,
		float3 center,
		float4 quaternion) {
	sq->radii = radii;
	sq->eps = eps;
	sq->pos = center;
	sq->quat = quaternion;
}

static bool run_case(
		const char* label,
		const Superquadric* sq,
		float sphere_radius,
		float3 sphere_center,
		float expected,
		float tolerance) {
	const float sdf = compute_superquadric_sphere_signed_distance(
			sq, sphere_radius, sphere_center);
	const float error = fabsf(sdf - expected);
	const bool pass = (error <= tolerance);

	std::printf(
			"%s\n"
			"  expected SDF: %+0.6f\n"
			"  computed SDF: %+0.6f\n"
			"  abs error   : %0.6f\n"
			"  result      : %s\n\n",
			label,
			expected,
			sdf,
			error,
			pass ? "PASS" : "FAIL");

	return pass;
}

static void record_case(
		bool pass,
		int* total_cases,
		int* passed_cases,
		bool* all_passed) {
	(*total_cases)++;
	if (pass) {
		(*passed_cases)++;
	}
	*all_passed = *all_passed && pass;
}

int main() {
	const float tol = 2.0e-3f;
	bool all_passed = true;
	int total_cases = 0;
	int passed_cases = 0;

	// Sphere-equivalent superquadric. Analytic baseline:
	// d = |c2 - c1| - (r_superquadric + r_sphere).
	Superquadric sphere_sq;
	const float sphere_sq_radius = 0.5f;
	const float sphere_radius = 0.4f;
	const float sphere_sum = sphere_sq_radius + sphere_radius;
	init_identity_superquadric_as_sphere(
			&sphere_sq, sphere_sq_radius, make_float3(0.0f, 0.0f, 0.0f));

	// 1) Separated case
	{
		const float center_distance = 2.0f;
		const float expected = center_distance - sphere_sum;
		record_case(run_case(
				"[Case 1] Separated",
				&sphere_sq,
				sphere_radius,
				make_float3(center_distance, 0.0f, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 2) Tangent case
	{
		const float center_distance = sphere_sum;
		const float expected = center_distance - sphere_sum;
		record_case(run_case(
				"[Case 2] Tangent",
				&sphere_sq,
				sphere_radius,
				make_float3(center_distance, 0.0f, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 3) Penetrating case
	{
		const float center_distance = 0.6f;
		const float expected = center_distance - sphere_sum;
		record_case(run_case(
				"[Case 3] Penetrating",
				&sphere_sq,
				sphere_radius,
				make_float3(center_distance, 0.0f, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 4) Concentric spheres
	{
		const float expected = -sphere_sum;
		record_case(run_case(
				"[Case 4] Concentric",
				&sphere_sq,
				sphere_radius,
				make_float3(0.0f, 0.0f, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 5) Separated along Y axis
	{
		const float center_distance = 2.0f;
		const float expected = center_distance - sphere_sum;
		record_case(run_case(
				"[Case 5] Separated Along Y",
				&sphere_sq,
				sphere_radius,
				make_float3(0.0f, center_distance, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 6) Tangent along diagonal direction
	{
		const float3 dir = normalize_float3(make_float3(1.0f, 1.0f, 1.0f));
		const float3 center = make_float3(
				dir.x * sphere_sum,
				dir.y * sphere_sum,
				dir.z * sphere_sum);
		record_case(run_case(
				"[Case 6] Tangent On Diagonal",
				&sphere_sq,
				sphere_radius,
				center,
				0.0f,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 7) Negative radius input should behave like positive radius.
	{
		const float center_distance = 2.0f;
		const float expected = center_distance - sphere_sum;
		record_case(run_case(
				"[Case 7] Negative Sphere Radius Input",
				&sphere_sq,
				-sphere_radius,
				make_float3(center_distance, 0.0f, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// Ellipsoid-like superquadric (eps = 1, 1). Along principal axes the exact
	// SDF remains analytic: distance to center minus axis radius minus sphere radius.
	Superquadric ellipsoid_sq;
	const float ellipsoid_sphere_radius = 0.3f;
	init_superquadric(
			&ellipsoid_sq,
			make_float3(1.0f, 0.5f, 0.25f),
			make_float2(1.0f, 1.0f),
			make_float3(0.0f, 0.0f, 0.0f),
			make_identity_quaternion());

	// 8) Ellipsoid separated along X.
	{
		const float center_distance = 2.0f;
		const float expected = center_distance - (1.0f + ellipsoid_sphere_radius);
		record_case(run_case(
				"[Case 8] Ellipsoid Separated Along X",
				&ellipsoid_sq,
				ellipsoid_sphere_radius,
				make_float3(center_distance, 0.0f, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 9) Ellipsoid tangent along Y.
	{
		const float center_distance = 0.5f + ellipsoid_sphere_radius;
		record_case(run_case(
				"[Case 9] Ellipsoid Tangent Along Y",
				&ellipsoid_sq,
				ellipsoid_sphere_radius,
				make_float3(0.0f, center_distance, 0.0f),
				0.0f,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 10) Ellipsoid penetrating along Z.
	{
		const float center_distance = 0.4f;
		const float expected = center_distance - (0.25f + ellipsoid_sphere_radius);
		record_case(run_case(
				"[Case 10] Ellipsoid Penetrating Along Z",
				&ellipsoid_sq,
				ellipsoid_sphere_radius,
				make_float3(0.0f, 0.0f, center_distance),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// Non-spherical exponents. Along principal axes the support location is still exact.
	Superquadric boxy_sq;
	const float boxy_sphere_radius = 0.2f;
	init_superquadric(
			&boxy_sq,
			make_float3(0.7f, 0.4f, 0.3f),
			make_float2(0.4f, 1.6f),
			make_float3(0.0f, 0.0f, 0.0f),
			make_identity_quaternion());

	// 11) Boxy superquadric separated along X.
	{
		const float center_distance = 1.4f;
		const float expected = center_distance - (0.7f + boxy_sphere_radius);
		record_case(run_case(
				"[Case 11] Boxy Separated Along X",
				&boxy_sq,
				boxy_sphere_radius,
				make_float3(center_distance, 0.0f, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 12) Boxy superquadric tangent along Z.
	{
		const float center_distance = 0.3f + boxy_sphere_radius;
		record_case(run_case(
				"[Case 12] Boxy Tangent Along Z",
				&boxy_sq,
				boxy_sphere_radius,
				make_float3(0.0f, 0.0f, center_distance),
				0.0f,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 13) Boxy superquadric penetrating along Y.
	{
		const float center_distance = 0.45f;
		const float expected = center_distance - (0.4f + boxy_sphere_radius);
		record_case(run_case(
				"[Case 13] Boxy Penetrating Along Y",
				&boxy_sq,
				boxy_sphere_radius,
				make_float3(0.0f, center_distance, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// Rotated ellipsoid: 90 degrees around Z swaps X/Y support radii.
	Superquadric rotated_ellipsoid_sq;
	init_superquadric(
			&rotated_ellipsoid_sq,
			make_float3(1.0f, 0.5f, 0.25f),
			make_float2(1.0f, 1.0f),
			make_float3(0.0f, 0.0f, 0.0f),
			make_z_rotation_quaternion(1.57079632679f));

	// 14) Rotated ellipsoid tangent along X (effective X radius becomes 0.5).
	{
		const float center_distance = 0.5f + ellipsoid_sphere_radius;
		record_case(run_case(
				"[Case 14] Rotated Ellipsoid Tangent Along X",
				&rotated_ellipsoid_sq,
				ellipsoid_sphere_radius,
				make_float3(center_distance, 0.0f, 0.0f),
				0.0f,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 15) Rotated ellipsoid separated along Y (effective Y radius becomes 1.0).
	{
		const float center_distance = 2.0f;
		const float expected = center_distance - (1.0f + ellipsoid_sphere_radius);
		record_case(run_case(
				"[Case 15] Rotated Ellipsoid Separated Along Y",
				&rotated_ellipsoid_sq,
				ellipsoid_sphere_radius,
				make_float3(0.0f, center_distance, 0.0f),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 16) Translation invariance for sphere-equivalent superquadric.
	{
		Superquadric shifted_sphere_sq;
		const float3 shifted_center = make_float3(1.0f, -2.0f, 0.5f);
		init_identity_superquadric_as_sphere(
				&shifted_sphere_sq, sphere_sq_radius, shifted_center);
		record_case(run_case(
				"[Case 16] Translated Sphere-Equivalent Tangent",
				&shifted_sphere_sq,
				sphere_radius,
				make_float3(
						shifted_center.x + sphere_sum,
						shifted_center.y,
						shifted_center.z),
				0.0f,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	// 17) Translation invariance for non-spherical principal-axis case.
	{
		Superquadric shifted_boxy_sq;
		const float3 shifted_center = make_float3(-1.5f, 0.75f, 0.25f);
		init_superquadric(
				&shifted_boxy_sq,
				make_float3(0.7f, 0.4f, 0.3f),
				make_float2(0.4f, 1.6f),
				shifted_center,
				make_identity_quaternion());
		const float center_distance = 0.45f;
		const float expected = center_distance - (0.4f + boxy_sphere_radius);
		record_case(run_case(
				"[Case 17] Translated Boxy Penetrating Along Y",
				&shifted_boxy_sq,
				boxy_sphere_radius,
				make_float3(
						shifted_center.x,
						shifted_center.y + center_distance,
						shifted_center.z),
				expected,
				tol),
				&total_cases,
				&passed_cases,
				&all_passed);
	}

	std::printf(
			"Summary: %d/%d cases passed\n"
			"Overall sanity check: %s\n",
			passed_cases,
			total_cases,
			all_passed ? "PASS" : "FAIL");
	return all_passed ? 0 : 1;
}