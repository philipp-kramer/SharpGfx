#include <optix.h>

#include "Programs\ray.cu"

extern "C" __global__ void __closesthit__radiance()
{
	UniformHitGroupData* hit_data = reinterpret_cast<UniformHitGroupData*>(optixGetSbtDataPointer());
	setPayload(hit_data->color);
}
