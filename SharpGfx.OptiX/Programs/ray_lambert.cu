#include <optix.h>
#include <cuda/helpers.h>

#include "Programs\ray.cu"

extern "C" __global__ void __closesthit__radiance()
{
    LambertHitGroupData* hit_data = reinterpret_cast<LambertHitGroupData*>(optixGetSbtDataPointer());
    const int primID = optixGetPrimitiveIndex();

    const ushort3 index =
        hit_data->indices == NULL
        ? ushort3 { 
            (unsigned short) (3 * primID + 0), 
            (unsigned short) (3 * primID + 1), 
            (unsigned short) (3 * primID + 2) }
        : hit_data->indices[primID];
    const float3 A = hit_data->vertices[index.x];
    const float3 B = hit_data->vertices[index.y];
    const float3 C = hit_data->vertices[index.z];

    const float3 objectN = cross(B - A, C - A);
    const float3 worldN = normalize(optixTransformNormalFromObjectToWorldSpace(objectN));

    const float t = optixGetRayTmax();
    float3 worldHitPosition = optixGetWorldRayOrigin() + t * optixGetWorldRayDirection();

    const auto lights_position = reinterpret_cast<float3*>(hit_data->lights_position);
    const auto light_colors = reinterpret_cast<float3*>(hit_data->lights_color);

    float3 color = hit_data->ambient_color;

    for (int i = 0; i < hit_data->point_lights_count; i++) {
        float3 lightDir = normalize(lights_position[i] - worldHitPosition);
        float cosTheta = fmax(dot(worldN, lightDir), 0.0f);
        float3 diffuse = cosTheta * light_colors[i];
        color = 1 - (1 - color) * (1 - diffuse);
    }

    float4 material_color = hit_data->material_color;
    setPayload(color * float3{ material_color.x, material_color.y, material_color.z });
}
