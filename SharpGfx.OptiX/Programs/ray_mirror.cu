#include <optix.h>
#include <cuda/helpers.h>

#include "Programs\ray.cu"

extern "C" __global__ void __closesthit__radiance()
{
    HitGroupData* hit_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const int primID = optixGetPrimitiveIndex();

    const ushort3 index =
        hit_data->indices == NULL
        ? ushort3{
            (unsigned short)(3 * primID + 0),
            (unsigned short)(3 * primID + 1),
            (unsigned short)(3 * primID + 2) }
    : hit_data->indices[primID];
    const float3 A = hit_data->vertices[index.x];
    const float3 B = hit_data->vertices[index.y];
    const float3 C = hit_data->vertices[index.z];

    const float3 objectN = normalize(cross(B - A, C - A));
    const float3 worldRayDir = optixGetWorldRayDirection();

    const float3 worldN = normalize(optixTransformVectorFromObjectToWorldSpace(objectN));
    const float3 worldReflectDir = reflect(worldRayDir, worldN);
    const float dp = dot(worldN, worldReflectDir);
    if (dp > 0)
    {
        const float t = optixGetRayTmax();
        float3 worldHitPosition = optixGetWorldRayOrigin() + t * worldRayDir;
        float3 color = trace(worldHitPosition, worldReflectDir);
        color.z += 1.0f;
        color.z *= 0.5f;
        setPayload(color);
    }
    else
    {
        const float3 objectRayDir = normalize(optixTransformVectorFromWorldToObjectSpace(worldRayDir));
        const float cosDN = 0.1f + .9f * fabsf(dot(objectRayDir, objectN));
        setPayload(float3{ 1, 0, 0 });
    }
}
