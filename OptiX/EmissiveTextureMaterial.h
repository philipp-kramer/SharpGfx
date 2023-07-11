#pragma once
#include <vector_types.h>

#include "graphic.h"
#include "Material.h"

class EmissiveTextureMaterial : public Material
{
public:
    cudaTextureObject_t texture;

    EmissiveTextureMaterial(OptixDeviceContext context, OptixPipelineCompileOptions& pipeline_compile_options, std::string rays_cu, cudaTextureObject_t texture)
        : Material(context, pipeline_compile_options, rays_cu)
    {
        this->texture = texture;
    }

    void SetRecordEntries(HitGroupData& record, Geometry& geometry) override
    {
        Material::SetRecordEntries(record, geometry);
        reinterpret_cast<UniformTextureHitGroupData&>(record).texture = texture;
        reinterpret_cast<UniformTextureHitGroupData&>(record).texcoords = reinterpret_cast<float2*>(geometry.d_texcoords);
    }
};
