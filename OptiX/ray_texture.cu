#include <optix.h>
#include <cuda/helpers.h>

#include "graphic.h"
#include "ray.cu"

extern "C" __global__ void __closesthit__radiance()
{
    UniformTextureHitGroupData* hit_data = reinterpret_cast<UniformTextureHitGroupData*>(optixGetSbtDataPointer());
    const int primID = optixGetPrimitiveIndex();

    const ushort3 index =
        hit_data->indices == NULL
        ? ushort3{
            (unsigned short)(3 * primID + 0),
            (unsigned short)(3 * primID + 1),
            (unsigned short)(3 * primID + 2) }
    : hit_data->indices[primID];

    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    const float r = (1.f - u - v);
    const float2 tc =
        r * hit_data->texcoords[index.x]
        + u * hit_data->texcoords[index.y]
        + v * hit_data->texcoords[index.z];

    float4 tex_color = tex2D<float4>(hit_data->texture, tc.x, tc.y);
    setPayload_0(tex_color.x);
    setPayload_1(tex_color.y);
    setPayload_2(tex_color.z);
}
