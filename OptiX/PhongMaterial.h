#pragma once
#include <vector_types.h>

#include "memory.h"
#include "Geometry.h"
#include "LambertMaterial.h"


class PhongMaterial : public LambertMaterial
{
public:
    float3 material_color_specular;
    float material_shininess;

    PhongMaterial(
        OptixDeviceContext context,
        OptixPipelineCompileOptions& pipeline_compile_options,
        std::string rays_cu,
        const float4 material_color,
        const float3 ambient_color,
        const float3* lights_position,
        const float3* lights_color,
        const int point_lights_count,
        const float3 material_color_specular,
        const float material_shininess)
        : LambertMaterial(
            context,
            pipeline_compile_options,
            rays_cu,
            material_color, 
            ambient_color,
            lights_position,
            lights_color,
            point_lights_count)
    {
        this->material_color_specular = material_color_specular;
        this->material_shininess = material_shininess;
    }

    void SetRecordEntries(HitGroupData& record, Geometry& geometry) override
    {
        LambertMaterial::SetRecordEntries(record, geometry);
        reinterpret_cast<PhongHitGroupData&>(record).material_color_specular = material_color_specular;
        reinterpret_cast<PhongHitGroupData&>(record).material_shininess = material_shininess;
    }
};
