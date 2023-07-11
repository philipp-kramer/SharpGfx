#pragma once
#include <optix_types.h>

struct Program
{
    std::vector<OptixProgramGroup> prog_groups{};
    std::vector<OptixProgramGroup> raygen_prog_group{};
    std::vector<OptixProgramGroup> radiance_miss_group{};
    //OptixProgramGroup              occlusion_miss_group = 0;
    std::vector<OptixProgramGroup> radiance_hit_group{};
    //OptixProgramGroup              occlusion_hit_group = 0;

    Program() {}

    Program(OptixDeviceContext context, std::vector<OptixModule> modules)
    {
        OptixProgramGroupOptions  program_group_options = {};
        char log[2048];

        const int count = (int) modules.size();
        prog_groups.resize(3 * count);

        raygen_prog_group.resize(count);
        radiance_miss_group.resize(count);
        radiance_hit_group.resize(count);

        int j = 0;
        for (int i = 0; i < count; i++)
        {
            createProgramGroup(context, OPTIX_PROGRAM_GROUP_KIND_RAYGEN, modules[i], "__raygen__rg", raygen_prog_group[i], program_group_options, log);
            prog_groups[j++] = raygen_prog_group[i];
        }
        for (int i = 0; i < count; i++)
        {
            createProgramGroup(context, OPTIX_PROGRAM_GROUP_KIND_MISS, modules[i], "__miss__radiance", radiance_miss_group[i], program_group_options, log);
            prog_groups[j++] = radiance_miss_group[i];
        }
        for (int i = 0; i < count; i++)
        {
            createProgramGroup(context, OPTIX_PROGRAM_GROUP_KIND_HITGROUP, modules[i], "__closesthit__radiance", radiance_hit_group[i], program_group_options, log);
            prog_groups[j++] = radiance_hit_group[i];
        }
    }

private:
    void createProgramGroup(OptixDeviceContext context, OptixProgramGroupKind kind, OptixModule ptx_module, const char* entryFunctionName, OptixProgramGroup& prog_group, OptixProgramGroupOptions program_group_options, char log[2048])
    {
        OptixProgramGroupDesc prog_group_desc = {};
        prog_group_desc.kind = kind;
        prog_group_desc.raygen.module = ptx_module;
        prog_group_desc.raygen.entryFunctionName = entryFunctionName;

        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &prog_group_desc,
            1,  // num program groups
            &program_group_options,
            log,
            &sizeof_log,
            &prog_group
        ));
    }
};
