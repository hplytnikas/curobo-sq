#include <cuda_runtime.h>
#include <math_constants.h>

#define SQ_NEWTON_ITERS 4

// --------------------------------------------
// Utility
// --------------------------------------------

__device__ __forceinline__
float signf_fast(float x)
{
    return copysignf(1.0f, x);
}

__device__ __forceinline__
float3 mul3(float3 a, float b)
{
    return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ __forceinline__
float3 sub3(float3 a, float3 b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ __forceinline__
float length3(float3 v)
{
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
}

// --------------------------------------------
// Superquadric parameter struct (SoA friendly)
// --------------------------------------------

struct Superquadric
{
    float3 center;

    float inv_a;
    float inv_b;
    float inv_c;

    float a;
    float b;
    float c;

    float e1;
    float e2;

    float k1; // 2/e2
    float k2; // e2/e1
    float k3; // 2/e1

    float outer_radius;
    float inner_radius;
};

// --------------------------------------------
// Superquadric implicit function
// --------------------------------------------

__device__
float superquadric_F(
    float3 p,
    const Superquadric& sq)
{
    float x = fabsf(p.x * sq.inv_a);
    float y = fabsf(p.y * sq.inv_b);
    float z = fabsf(p.z * sq.inv_c);

    float u = __powf(x, sq.k1);
    float v = __powf(y, sq.k1);

    float A = u + v;

    float t1 = __powf(A, sq.k2);
    float t2 = __powf(z, sq.k3);

    return t1 + t2 - 1.0f;
}

// --------------------------------------------
// Analytic gradient
// --------------------------------------------

__device__
float3 superquadric_grad(
    float3 p,
    const Superquadric& sq)
{
    float x = p.x * sq.inv_a;
    float y = p.y * sq.inv_b;
    float z = p.z * sq.inv_c;

    float ax = fabsf(x);
    float ay = fabsf(y);
    float az = fabsf(z);

    float u = __powf(ax, sq.k1);
    float v = __powf(ay, sq.k1);

    float A = u + v;

    float Ap = __powf(A, sq.k2 - 1.0f);

    float gx =
        sq.k1 * sq.k2 *
        signf_fast(x) *
        __powf(ax, sq.k1 - 1.0f) *
        Ap * sq.inv_a;

    float gy =
        sq.k1 * sq.k2 *
        signf_fast(y) *
        __powf(ay, sq.k1 - 1.0f) *
        Ap * sq.inv_b;

    float gz =
        sq.k3 *
        signf_fast(z) *
        __powf(az, sq.k3 - 1.0f) *
        sq.inv_c;

    return make_float3(gx, gy, gz);
}

// --------------------------------------------
// Radial projection solve
// --------------------------------------------

__device__
float radial_lambda(
    float3 p,
    const Superquadric& sq)
{
    float lambda = 1.0f;

    #pragma unroll
    for(int i=0;i<SQ_NEWTON_ITERS;i++)
    {
        float3 q = mul3(p, lambda);

        float f = superquadric_F(q, sq);

        float3 g = superquadric_grad(q, sq);

        float df =
            g.x * p.x +
            g.y * p.y +
            g.z * p.z;

        lambda -= f / df;
    }

    return lambda;
}

// --------------------------------------------
// Sphere distance
// --------------------------------------------

__device__
float superquadric_sphere_distance(
    float3 sphere_center,
    float sphere_radius,
    const Superquadric& sq)
{
    float3 p = sub3(sphere_center, sq.center);

    float dist_center = length3(p);

    // Early rejection
    if(dist_center > sq.outer_radius + sphere_radius)
        return dist_center - sq.outer_radius - sphere_radius;

    // Early inside
    if(dist_center < sq.inner_radius - sphere_radius)
        return -(sq.inner_radius - sphere_radius - dist_center);

    float lambda = radial_lambda(p, sq);

    float3 s = mul3(p, lambda);

    float dist = length3(sub3(p, s));

    return dist - sphere_radius;
}

// --------------------------------------------
// Batched kernel
// --------------------------------------------

__global__
void superquadric_sphere_batch(
    const Superquadric* sqs,
    const float3* sphere_centers,
    const float* sphere_radii,
    float* distances,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= N)
        return;

    const Superquadric& sq = sqs[idx];

    float3 c = sphere_centers[idx];
    float r = sphere_radii[idx];

    distances[idx] =
        superquadric_sphere_distance(c, r, sq);
}