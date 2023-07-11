#pragma once
#include <vector_types.h>

#include "graphic.h"
#include "Material.h"

class EmissiveMaterial : public Material
{
public:
    float3 color{};

    EmissiveMaterial(OptixDeviceContext context, OptixPipelineCompileOptions& pipeline_compile_options, std::string rays_cu, const float3 color)
        : Material(context, pipeline_compile_options, rays_cu)
    {
        this->color = color;
    }

    void SetRecordEntries(HitGroupData& record, Geometry& geometry) override
    {
        Material::SetRecordEntries(record, geometry);
        reinterpret_cast<UniformHitGroupData&>(record).color = color;
    }
};
