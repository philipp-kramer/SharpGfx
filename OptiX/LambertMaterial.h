#pragma once
#include <vector_types.h>

#include "graphic.h"
#include "Material.h"

class LambertMaterial : public Material
{
public:
    float4      material_color{};
    float3      ambient_color{};
    CUdeviceptr lights_position; // TODO: destroy
    CUdeviceptr lights_color; // TODO: destroy
    int         point_lights_count;

    LambertMaterial(
        OptixDeviceContext context, 
        OptixPipelineCompileOptions& pipeline_compile_options, 
        const std::string rays_cu,
        const const float4 material_color,
        const float3 ambient_color, 
        const float3* lights_position,
        const float3* lights_color,
        const int point_lights_count)
        : Material(context, pipeline_compile_options, rays_cu)
    {
        this->material_color = material_color;
        this->ambient_color = ambient_color;
        this->lights_position = copy_array_to_device<float3>(point_lights_count, lights_position);
        this->lights_color = copy_array_to_device<float3>(point_lights_count, lights_color);
        this->point_lights_count = point_lights_count;
    }

    void SetRecordEntries(HitGroupData& record, Geometry& geometry) override
    {
        Material::SetRecordEntries(record, geometry);
        reinterpret_cast<LambertHitGroupData&>(record).material_color = material_color;
        reinterpret_cast<LambertHitGroupData&>(record).ambient_color = ambient_color;
        reinterpret_cast<LambertHitGroupData&>(record).lights_position = lights_position;
        reinterpret_cast<LambertHitGroupData&>(record).lights_color = lights_color;
        reinterpret_cast<LambertHitGroupData&>(record).point_lights_count = point_lights_count;
    }
};
