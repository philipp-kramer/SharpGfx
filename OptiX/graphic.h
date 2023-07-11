#pragma once

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    //RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    OptixTraversableHandle handle;

    Params* d_params;
};


struct RayGenData
{
};


struct MissData
{
    float3 bg_color;
};


class HitGroupData
{
public:
    float3* vertices;
    ushort3* indices;

    virtual ~HitGroupData() {}
};

class UniformHitGroupData : public HitGroupData // TODO: rename to Emissive
{
public:
    float3  color;

    ~UniformHitGroupData() override { }
};

class UniformTextureHitGroupData : public HitGroupData  // TODO: rename to Emissive
{
public:
    float3  color;
    cudaTextureObject_t texture;
    float2* texcoords;
};

class LambertHitGroupData : public HitGroupData
{
public:
    float4      material_color;
    float3      ambient_color;
    CUdeviceptr lights_position;
    CUdeviceptr lights_color;
    int         point_lights_count;
};

class LambertTextureHitGroupData : public LambertHitGroupData
{
public:
    cudaTextureObject_t texture;
    float2* texcoords;
};

class PhongHitGroupData : public LambertHitGroupData
{
public:
    float3 material_color_specular;
    float material_shininess;
};

