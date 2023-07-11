#pragma once
#include <string>
#include <nvrtc.h>
#include <optix_stubs.h>
#include <optix.h>
#include <vector>

#include "Exception.h"


#define NVRTC_CHECK_ERROR(func)     \
    do                              \
    {                               \
        nvrtcResult code = func;    \
        if (code != NVRTC_SUCCESS)  \
            throw std::runtime_error(std::string(nvrtcGetErrorString(code))); \
    } while(0)
// "ERROR: " __FILE__ "(" LINE_STR "): " + std::string(nvrtcGetErrorString(code

class Material
{
public:
    OptixModule ptx_module;

    Material(OptixDeviceContext context, OptixPipelineCompileOptions& pipeline_compile_options, std::string rays_cu)
    {
        ptx_module = createModule(context, pipeline_compile_options, rays_cu);
    }

    virtual void SetRecordEntries(HitGroupData& record, Geometry& geometry)
    {
        record.vertices = reinterpret_cast<float3*>(geometry.d_vertices); // depends on triangle_input.triangleArray.vertexFormat
        record.indices = reinterpret_cast<ushort3*>(geometry.d_indices);
    }

private:
    OptixModule createModule(OptixDeviceContext context, OptixPipelineCompileOptions& pipeline_compile_options, const std::string rays_cu)
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

        char   log[2048];
        size_t sizeof_log = sizeof(log);

        std::string ptx = getPtx(rays_cu, "OptiX");
        const char* ptxString = const_cast<const char*>(ptx.c_str());

        OptixModule ptx_module;
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            ptxString,
            ptx.size(),
            log,
            &sizeof_log,
            &ptx_module
        ));
        return ptx_module;
    }

    std::string getPtx(const std::string rays_cu, const std::string name)
    {
        std::string ptx;
        std::string location;

        // Create program
        nvrtcProgram prog = 0;
        NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, rays_cu.c_str(), name.c_str(), 0, NULL, NULL));

        std::vector<const char*> options;
        std::string optix_install_dir = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0/";
        std::string cuda_install_dir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/";
        std::vector<const char*> compiler_options;
        compiler_options.push_back("-std=c++11");
        compiler_options.push_back("-arch=compute_75");
        compiler_options.push_back("-use_fast_math");
        compiler_options.push_back("-lineinfo");
        compiler_options.push_back("-default-device");
        compiler_options.push_back("-rdc=true");
        compiler_options.push_back("-D=__x86_64");
        compiler_options.push_back("-I=..");
        std::vector<std::string> dirs;
        dirs.push_back(std::string("-I=") + optix_install_dir + std::string("include"));
        dirs.push_back(std::string("-I=") + optix_install_dir + std::string("SDK/cuda"));
        dirs.push_back(std::string("-I=") + optix_install_dir + std::string("SDK"));
        dirs.push_back(std::string("-I=") + cuda_install_dir + std::string("include"));
        for (size_t i = 0; i < dirs.size(); i++)
        {
            compiler_options.push_back(dirs[i].c_str());
        }

        const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int) compiler_options.size(), compiler_options.data());

        if (compileRes != NVRTC_SUCCESS) {
            std::string nvrtcLog = std::string();
            size_t log_size = 0;
            NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
            nvrtcLog.resize(log_size);
            if (log_size > 1)
            {
                NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &nvrtcLog[0]));
            }
            std::cerr << nvrtcLog << "\n";
            throw std::runtime_error("NVRTC Compilation failed.\n" + nvrtcLog);
        }

        // Retrieve PTX code
        size_t ptx_size = 0;
        NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
        ptx.resize(ptx_size);
        NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

        // Cleanup
        NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));

        return ptx;
    }
};
