#include <array>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <string>

#include "Camera.h"
#include "CUDAOutputBuffer.h"
#include "window.h"
#include "Exception.h"
#include "Geometry.h"
#include "Instance.h"
#include "graphic.h"
#include "memory.h"
#include "optix_wrapper.h"
#include "EmissiveMaterial.h"
#include "EmissiveTextureMaterial.h"
#include "LambertMaterial.h"
#include "LambertTextureMaterial.h"
#include "PhongMaterial.h"


template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<LambertTextureHitGroupData> MaxHitGroupRecord; // TODO: export functionality

const int32_t samples_per_launch = 16;


unsigned int indexOf(Geometry** geometries, int count, OptixTraversableHandle handle)
{
    for (int i = 0; i < count; i++)
    {
        if (geometries[i]->gas_handle == handle) return i;
    }
    throw std::runtime_error("instance geometry not defined");
}

OptixTraversableHandle buildAccelStructure(CudaState& cuState, Geometry** geometries, const int b_count, OptixInstance** instances, const int i_count)
{
    int size = i_count * sizeof(OptixInstance);

    auto flatInstances = std::vector<OptixInstance>(i_count);
    for (size_t i = 0; i < i_count; i++)
    {
        const auto instance = instances[i];
        flatInstances[i] = *instance;
        flatInstances[i].instanceId = (unsigned int)i;
        flatInstances[i].sbtOffset = indexOf(geometries, b_count, instance->traversableHandle); // TODO: material
    }

    if (cuState.dInstanceAcceleratorBuffer == 0)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cuState.dOptixInstances), size));

        cuState.instanceBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        cuState.instanceBuildInput.instanceArray.instances = cuState.dOptixInstances;
        cuState.instanceBuildInput.instanceArray.numInstances = i_count;

        cuState.acceleratorBuildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        cuState.acceleratorBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            cuState.context,
            &cuState.acceleratorBuildOptions,
            &cuState.instanceBuildInput,
            1u,
            &cuState.acceleratorBufferSizes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cuState.dInstanceAcceleratorBuffer), cuState.acceleratorBufferSizes.outputSizeInBytes));
    }
    else 
    {
        cuState.acceleratorBuildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
    }

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(cuState.dOptixInstances), flatInstances.data(), size, cudaMemcpyHostToDevice));

    CUdeviceptr dTempBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBuffer), cuState.acceleratorBufferSizes.tempSizeInBytes));

    OptixTraversableHandle instanceAcceleratorHandle{ 0 };
    OPTIX_CHECK(optixAccelBuild(
        cuState.context,
        0,
        &cuState.acceleratorBuildOptions,
        &cuState.instanceBuildInput,
        1,
        dTempBuffer,
        cuState.acceleratorBufferSizes.tempSizeInBytes,
        cuState.dInstanceAcceleratorBuffer,
        cuState.acceleratorBufferSizes.outputSizeInBytes,
        &instanceAcceleratorHandle,
        nullptr,
        0));

    CUDA_CHECK(cudaFree((void*) dTempBuffer));

    return instanceAcceleratorHandle;
}

int update(Context& context, Geometry** geometries, const int b_count, OptixInstance** instances, const int i_count)
{
    try
    {
        buildAccelStructure(context.cuState, geometries, b_count, instances, i_count);
        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return -1;
    }
}


OptixPipeline createPipeline(OptixDeviceContext context, Program& program, const OptixPipelineCompileOptions& pipeline_compile_options)
{
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char   log[2048];
    size_t sizeof_log = sizeof(log);
    OptixPipeline pipeline;
    OPTIX_CHECK_LOG(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program.prog_groups.data(),
        (int) program.prog_groups.size(),
        log,
        &sizeof_log,
        &pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    for (size_t i = 0; i < program.prog_groups.size(); i++)
    {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program.prog_groups[i], &stack_sizes));
    }

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 2;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));

    return pipeline;
}


OptixShaderBindingTable createSBT(Program& program, Geometry** geometries, Material** materials, const int geometries_count, const float3 bg_color)
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(program.raygen_prog_group[0], &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));
    MissRecord ms_sbt[1];
    OPTIX_CHECK(optixSbtRecordPackHeader(program.radiance_miss_group[0], &ms_sbt[0]));
    ms_sbt[0].data.bg_color = bg_color;
    //OPTIX_CHECK(optixSbtRecordPackHeader(program.occlusion_miss_group, &ms_sbt[1]));
    //ms_sbt[1].data.bg_color = make_float3(0.0f);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_records), ms_sbt, miss_record_size * RAY_TYPE_COUNT, cudaMemcpyHostToDevice));

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(MaxHitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), hitgroup_record_size * RAY_TYPE_COUNT * geometries_count));

    auto hitgroup_records = std::vector<MaxHitGroupRecord>(RAY_TYPE_COUNT * geometries_count);
    for (int i = 0; i < geometries_count; ++i)
    {
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

            OPTIX_CHECK(optixSbtRecordPackHeader(program.radiance_hit_group[i], &hitgroup_records[sbt_idx]));
            materials[i]->SetRecordEntries(hitgroup_records[sbt_idx].data, *geometries[i]);
        }
        {
            //const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
            //memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);
            //OPTIX_CHECK(optixSbtRecordPackHeader(cuState.occlusion_hit_group, &hitgroup_records[sbt_idx]));
        }
    }

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records), hitgroup_records.data(), hitgroup_record_size * RAY_TYPE_COUNT * geometries_count, cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt{};
    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_records;
    sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    sbt.missRecordCount = RAY_TYPE_COUNT;
    sbt.hitgroupRecordBase = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    sbt.hitgroupRecordCount = RAY_TYPE_COUNT * geometries_count;

    return sbt;
}


Params initLaunchParams()
{
    Params params{};

    params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    params.samples_per_launch = samples_per_launch;
    params.subframe_index = 0u;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.d_params), sizeof(Params)));
    return params;
}


void handleCameraUpdate(Camera& camera, Params& params)
{
    camera.changed = false;

    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}


void handleResize(Window& window, Camera& camera, CudaState& cuState, Params& params)
{
    if (!window.resizeDirty)
        return;
    window.resizeDirty = false;
    camera.changed = true;

    params.width = window.width;
    params.height = window.height;

    cuState.output_buffer->resize(params.width, params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.accum_buffer), params.width * params.height * sizeof(float4)));
}


void updateState(Window& window, Camera& camera, Context& context)
{
    // Update params on device
    if (window.resizeDirty)
    {
        handleResize(window, camera, context.cuState, context.params);
    }
    if (camera.changed)
    {
        context.params.subframe_index = 0;
        handleCameraUpdate(camera, context.params);
    }
}


void launchSubframe(CudaState& cuState, Params& params)
{
    uchar4* result_buffer_data = cuState.output_buffer->map();
    params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.d_params),
        &params, sizeof(Params),
        cudaMemcpyHostToDevice, cuState.stream
    ));

    OPTIX_CHECK(optixLaunch(
        cuState.pipeline,
        cuState.stream,
        reinterpret_cast<CUdeviceptr>(params.d_params),
        sizeof(Params),
        &cuState.sbt,
        params.width,
        params.height,
        1 // depth
    ));
    cuState.output_buffer->unmap();
    CUDA_SYNC_CHECK();
}


void cleanupState(Context context)
{
    delete context.cuState.output_buffer;
    CUDA_CHECK(cudaFree((void*)context.cuState.dOptixInstances));
    OPTIX_CHECK(optixPipelineDestroy(context.cuState.pipeline));
    
    for (size_t i = 0; i < context.program.prog_groups.size(); i++)
    {
        OPTIX_CHECK(optixProgramGroupDestroy(context.program.prog_groups[i]));
    }
    OPTIX_CHECK(optixDeviceContextDestroy(context.cuState.context));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(context.cuState.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(context.cuState.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(context.cuState.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(context.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(context.params.d_params)));
}


int build(Context& context, Geometry** geometries, Material** materials, const int b_count, OptixInstance** instances, const int i_count, const float* bg_color)
{
    try
    {
        OptixTraversableHandle handle = buildAccelStructure(context.cuState, geometries, b_count, instances, i_count);

        const auto cuContext = context.cuState.context;
        auto modules = std::vector<OptixModule>(b_count);
        for (size_t i = 0; i < b_count; i++)
        {
            modules[i] = materials[i]->ptx_module;
        }
        Program program(cuContext, modules);
        context.program = program;
        context.cuState.pipeline = createPipeline(cuContext, program, context.pipeline_compile_options);
        context.cuState.sbt = createSBT(program, geometries, materials, b_count, *reinterpret_cast<const float3*>(bg_color));

        if (cuContext != 0) {
            context.params = initLaunchParams();
            context.params.handle = handle;
            CUDA_CHECK(cudaStreamCreate(&context.cuState.stream));
            return 0;
        } else {
            std::cerr << "failed to create cuda context\n";
            return -2;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return -1;
    }
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}


OptixDeviceContext createContext()
{
    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    if (_DEBUG)
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    }
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    return context;
}

Context* create_context()
{
    try
    {
        CUDA_CHECK(cudaFree(0)); // Initialize CUDA
    }
    catch (const Exception ex)
    {
        std::printf(ex.what());
    }
    Context* context = new Context();
    context->cuState.context = createContext();
    return context;
}

Material* Material_Create(Context context, const char* rays_cu)
{
    return new Material(
        context.cuState.context, 
        context.pipeline_compile_options, 
        std::string(rays_cu));
}

Material* EmissiveMaterial_Create(Context context, const char* rays_cu, const float* color3)
{
    return new EmissiveMaterial(
        context.cuState.context,
        context.pipeline_compile_options,
        std::string(rays_cu),
        *reinterpret_cast<const float3*>(color3));
}

Material* EmissiveTextureMaterial_Create(Context context, const char* rays_cu, Texture* texture)
{
    const auto cudaTexture = texture->createCudaTexture().object; // TODO: dispose
    return new EmissiveTextureMaterial(
        context.cuState.context,
        context.pipeline_compile_options,
        std::string(rays_cu),
        cudaTexture);
}

Material* LambertMaterial_Create(Context context, const char* rays_cu, const float* material_color, const float* ambient_color, const float* light_positions, const float* light_colors, const int point_lights_count)
{
    return new LambertMaterial(
        context.cuState.context,
        context.pipeline_compile_options,
        std::string(rays_cu),
        *reinterpret_cast<const float4*>(material_color),
        *reinterpret_cast<const float3*>(ambient_color),
        reinterpret_cast<const float3*>(light_positions),
        reinterpret_cast<const float3*>(light_colors),
        point_lights_count);
}

Material* LambertTextureMaterial_Create(Context context, const char* rays_cu, Texture* texture, const float* ambient_color, const float* light_positions, const float* light_colors, const int point_lights_count)
{
    const auto cudaTexture = texture->createCudaTexture().object; // TODO: dispose
    return new LambertTextureMaterial(
        context.cuState.context,
        context.pipeline_compile_options,
        std::string(rays_cu),
        cudaTexture,
        *reinterpret_cast<const float3*>(ambient_color),
        reinterpret_cast<const float3*>(light_positions),
        reinterpret_cast<const float3*>(light_colors),
        point_lights_count);
}

Material* PhongMaterial_Create(
    Context context,
    const char* rays_cu,
    const float* material_color,
    const float* ambient_color,
    const float* lights_position,
    const float* lights_color,
    const int point_lights_count,
    const float* material_color_specular,
    const float material_shininess)
{
    return new PhongMaterial(
        context.cuState.context,
        context.pipeline_compile_options,
        std::string(rays_cu),
        *reinterpret_cast<const float4*>(material_color),
        *reinterpret_cast<const float3*>(ambient_color),
        reinterpret_cast<const float3*>(lights_position),
        reinterpret_cast<const float3*>(lights_color),
        point_lights_count,
        *reinterpret_cast<const float3*>(material_color_specular),
        material_shininess);
}

void Material_Destroy(Material* material)
{
    try
    {
        OPTIX_CHECK(optixModuleDestroy(material->ptx_module));
    }
    catch (const Exception ex)
    {
        std::printf(ex.what());
    }

    delete material;
}


Window* open_window(Context& context, const char* title, int width, int height)
{
    context.params.width = width;
    context.params.height = height;

    try
    {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&context.params.accum_buffer),
            width * height * sizeof(float4)
        ));
    }
    catch (const Exception ex)
    {
        std::printf(ex.what());
    }

    Window* window;
    try
    {
        window = new Window(title, width, height);

        context.cuState.output_buffer = new CUDAOutputBuffer<uchar4>(CUDAOutputBufferType::CUDA_DEVICE, context.params.width, context.params.height);
        context.cuState.output_buffer->setStream(context.cuState.stream);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return nullptr;
    }
    return window;
}

int render(Window& window, Camera& camera, Context& context)
{
    try
    {
        glfwPollEvents();
        updateState(window, camera, context);
        launchSubframe(context.cuState, context.params);
        window.displaySubframe(context.cuState.output_buffer->getPBO());

        ++context.params.subframe_index;
        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return -1;
    }
}

int close(Window& window, Camera& camera, Context& context)
{
    try
    {
        CUDA_SYNC_CHECK();
        cleanupState(context);
        delete &camera;
        delete &window; // TODO: figure out why deleting window throws exception when running native
		delete &context;
        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return -1;
    }
}


void readFile(std::string& str, const std::string& file_path)
{
    std::ifstream file(file_path.c_str(), std::ios::binary);
    if (file.good())
    {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
    }
    else
    {
        std::string err = "Couldn't open source file " + file_path;
        throw std::runtime_error(err.c_str());
    }
}

int main(int argc, char* argv[])
{
    printf("OptiX version %d\r\n", OPTIX_VERSION);

    try
    {
        float tex_coords[] = {
             0.f, 0.f,
             1.f, 1.f,
             1.f, 0.f,
             0.f, 1.f
        };

        unsigned short idxs[] = { 0, 1, 2, 0, 1, 3 };
        float planet_vertices[] = {
            -0.5f,-0.5f,-.1f,
             0.5f, 0.5f,-.1f,
             0.5f,-0.5f,-.1f,
            -0.5f, 0.5f,-.1f,
        };
        float triangle_vertices[] = {
             0.5f, 0.5f, .0f,
             0.0f,-0.5f, .0f,
            -0.5f, 0.5f, -.2f
        };

        Context context = *create_context();
        Geometry* geometries[] =
        {
            ShortIndexedGeometry_Create(
                context,
                &planet_vertices[0], 
                tex_coords,
                4,
                &idxs[0], 2),
            Geometry_Create(context, &triangle_vertices[0], nullptr, 3)
        };

        float identity[12] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f };
        OptixInstance* instances[] =
        {
            Instance_Create(*geometries[0], identity),
            Instance_Create(*geometries[1], identity)
        };

        Texture* texture = new Texture("C:/Users/pkramer/_source/svn/dev/trunk/CG/Lecture/Demo/Resources/Images/earth.jpg");

        std::string texture_rays_cu;
        readFile(texture_rays_cu, "ray_texture.cu");
        Material* textureMaterial = EmissiveTextureMaterial_Create(context, texture_rays_cu.data(), texture);

        std::string uniform_rays_cu;
        readFile(uniform_rays_cu, "ray_uniform.cu");
        const float color[3] = { 1, 0, 0 };
        Material* uniformMaterial = EmissiveMaterial_Create(context, uniform_rays_cu.data(), color);

        Material* materials[2] = { textureMaterial, uniformMaterial };

        int error = build(context, geometries, materials, 2, instances, 2, new float[3] { 0.f, 0.f, 0.4f });
        if (error < 0) return error;

        Window window = *open_window(context, "test", 800, 600);

        Camera camera = *create_camera();
        int i = 0;
        do {
            error = render(window, camera, context);

            float translation[12] = {
                1.0f, 0.0f, 0.0f, i++ * 0.00001f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f };            
            Instance_Update(instances[1], translation);
            
            update(context, geometries, 2, instances, 2);
        } while (error == 0 && !window_should_close(window));

        Material_Destroy(textureMaterial);
        Material_Destroy(uniformMaterial);
        Texture_Destroy(texture);
        for (size_t i = 0; i < 2; i++)
        {
            Geometry_Destroy(geometries[i]);
        }
        close(window, camera, context);
        return error;
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return -3;
    }
}
