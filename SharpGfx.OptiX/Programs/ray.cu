//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <cuda/helpers.h>

#include "graphic.h"

extern "C" {
    __constant__ Params params;
}


static __forceinline__ __device__ void setPayload_0(float f)
{
    optixSetPayload_0(float_as_int(f));
}

static __forceinline__ __device__ void setPayload_1(float f)
{
    optixSetPayload_1(float_as_int(f));
}

static __forceinline__ __device__ void setPayload_2(float f)
{
    optixSetPayload_2(float_as_int(f));
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    setPayload_0(p.x);
    setPayload_1(p.y);
    setPayload_2(p.z);
}


static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const float2 d = 2.0f * make_float2(
        static_cast<float>(idx.x) / static_cast<float>(dim.x),
        static_cast<float>(idx.y) / static_cast<float>(dim.y))
        - 1.0f;

    origin = params.eye;
    direction = normalize(d.x * U + d.y * V + W);
}


// Trace the ray against our scene hierarchy
static __forceinline__ __device__ float3 trace(float3 ray_origin, float3 ray_direction)
{
    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.1f,                // Min intersection distance
        1e16f,               // Max intersection distance
        0.0f,                // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        1,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1, p2);

    float3 result;
    result.x = int_as_float(p0);
    result.y = int_as_float(p1);
    result.z = int_as_float(p2);

    return result;
}
// TODO: first sbt data shader

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera location through the screen
    float3 ray_origin, ray_direction;
    computeRay(idx, dim, ray_origin, ray_direction);

    float3 result = trace(ray_origin, ray_direction);

    // Record results in our output raster
    uchar4 color = {
        (unsigned char)(255 * result.x),
        (unsigned char)(255 * result.y),
        (unsigned char)(255 * result.z),
        255
    };
    params.frame_buffer[idx.y * params.width + idx.x] = color;
}


extern "C" __global__ void __miss__radiance()
{
    MissData* miss_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    setPayload(miss_data->bg_color);
}


