#pragma once
#include <vector_types.h>

#include "graphic.h"
#include "Material.h"

class LambertTextureMaterial : public Material
{
public:
    cudaTextureObject_t texture;
    float3      ambient_color{};
    CUdeviceptr lights_position; // TODO: destroy
    CUdeviceptr lights_color; // TODO: destroy
    int         point_lights_count;

    LambertTextureMaterial(
        OptixDeviceContext context,
        OptixPipelineCompileOptions& pipeline_compile_options,
        const std::string rays_cu,
        const cudaTextureObject_t texture,
        const float3 ambient_color,
        const float3* light_positions,
        const float3* light_colors,
        const int point_lights_count)
        : Material(context, pipeline_compile_options, rays_cu)
    {
        this->texture = texture;
        this->ambient_color = ambient_color;
        this->lights_position = copy_array_to_device<float3>(point_lights_count, light_positions);
        this->lights_color = copy_array_to_device<float3>(point_lights_count, light_colors);
        this->point_lights_count = point_lights_count;
    }

    void SetRecordEntries(HitGroupData& record, Geometry& geometry) override
    {
        Material::SetRecordEntries(record, geometry);
        reinterpret_cast<LambertTextureHitGroupData&>(record).texture = texture;
        reinterpret_cast<LambertTextureHitGroupData&>(record).texcoords = reinterpret_cast<float2*>(geometry.d_texcoords);
        reinterpret_cast<LambertTextureHitGroupData&>(record).ambient_color = ambient_color;
        reinterpret_cast<LambertTextureHitGroupData&>(record).lights_position = lights_position;
        reinterpret_cast<LambertTextureHitGroupData&>(record).lights_color = lights_color;
        reinterpret_cast<LambertTextureHitGroupData&>(record).point_lights_count = point_lights_count;
    }
};
