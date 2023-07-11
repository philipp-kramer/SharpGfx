#pragma once
#include "memory.h"
#include "context.h"


// TODO: support multiple meshes and textures per geometry as in optix7course
class Geometry
{
public:
	const CUdeviceptr d_vertices;
	const CUdeviceptr d_texcoords;
	const CUdeviceptr d_indices;
	OptixTraversableHandle gas_handle; // gas = global acceleration structure

	Geometry(
		Context& context,
		const float* vertices, 
		const float* texcoords,
		const int vertices_count) :
		d_vertices(copy_array_to_device<float>(3 * vertices_count, vertices)),
		d_texcoords(texcoords == nullptr ? NULL : copy_array_to_device<float2>(vertices_count, (const float2*) texcoords)),
		d_indices(NULL),
		vertices_count(vertices_count),
		triangle_count(0),
		indexFormat(OPTIX_INDICES_FORMAT_NONE),
		indexStrideInBytes(0)
	{
		build_geometry(context);
	}

	Geometry(
		Context& context,
		const float* vertices, 
		const float* texcoords,
		const int vertices_count,
		const unsigned int* indices, 
		const int triangle_count) :
		d_vertices(copy_array_to_device<float>(3 * vertices_count, vertices)),
		d_texcoords(texcoords == nullptr ? NULL : copy_array_to_device<float2>(vertices_count, (const float2*)texcoords)),
		d_indices(copy_array_to_device<unsigned int>(3 * triangle_count, indices)),
		vertices_count(vertices_count),
		triangle_count(triangle_count),
		indexFormat(OPTIX_INDICES_FORMAT_UNSIGNED_INT3),
		indexStrideInBytes(sizeof(uint3))
	{
		build_geometry(context);
	}

	Geometry(
		Context& context,
		const float* vertices,
		const float* texcoords,
		const int vertices_count,
		const unsigned short* indices, 
		const int triangle_count) :
		d_vertices(copy_array_to_device<float>(3 * vertices_count, vertices)),
		d_texcoords(texcoords == nullptr ? NULL : copy_array_to_device<float2>(vertices_count, (const float2*)texcoords)),
		d_indices(copy_array_to_device<unsigned short>(3 * triangle_count, indices)),
		vertices_count(vertices_count),
		triangle_count(triangle_count),
		indexFormat(OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3),
		indexStrideInBytes(sizeof(ushort3))
	{
		build_geometry(context);
	}

	~Geometry()
	{
		// unchecked freeing of memory in destructor
		cudaFree(reinterpret_cast<void*>(d_gas_output_buffer));
		cudaFree(reinterpret_cast<void*>(d_vertices));
		if (d_texcoords != NULL) cudaFree(reinterpret_cast<void*>(d_texcoords));
		if (d_indices != NULL) cudaFree(reinterpret_cast<void*>(d_indices));
	}

private:
	const int vertices_count;
	const int triangle_count;
	const OptixIndicesFormat indexFormat;
	const unsigned int indexStrideInBytes;
	CUdeviceptr d_gas_output_buffer = 0;

	void Geometry::triangle_input(OptixBuildInput& triangle_input)
	{
		triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.numVertices = vertices_count;
		triangle_input.triangleArray.vertexBuffers = &d_vertices;

		triangle_input.triangleArray.indexFormat = indexFormat;
		triangle_input.triangleArray.numIndexTriplets = triangle_count;
		triangle_input.triangleArray.indexBuffer = d_indices;
		triangle_input.triangleArray.indexStrideInBytes = indexStrideInBytes;

		uint32_t* triangle_input_flags = new uint32_t[1];
		triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;
		triangle_input.triangleArray.flags = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = 1;
	}

	void build_geometry(Context& context)
	{
		auto cuState = context.cuState;

		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixBuildInput build_input{};
		triangle_input(build_input);
		OptixAccelBufferSizes bufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			cuState.context,
			&accelOptions,
			&build_input,
			1,  // num_build_inputs
			&bufferSizes
		));

		// prepare compaction

		CUdeviceptr compactedSizeBuffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedSizeBuffer), sizeof(uint64_t)));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer;

		// build 

		CUdeviceptr d_temp_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), bufferSizes.tempSizeInBytes));

		CUdeviceptr d_temp_buffer_output_gas;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_output_gas), bufferSizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(
			cuState.context,
			cuState.stream,
			&accelOptions,
			&build_input,
			1,  // num_build_inputs
			d_temp_buffer,
			bufferSizes.tempSizeInBytes,
			d_temp_buffer_output_gas,
			bufferSizes.outputSizeInBytes,
			&gas_handle,
			&emitDesc, 1
		));
		CUDA_SYNC_CHECK();

		// perform compaction

		uint64_t compactedSize;
		CUDA_CHECK(cudaMemcpy(&compactedSize, reinterpret_cast<void**>(compactedSizeBuffer), sizeof(uint64_t), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compactedSize));
		OPTIX_CHECK(optixAccelCompact(
			cuState.context,
			cuState.stream,
			gas_handle,
			d_gas_output_buffer,
			compactedSize,
			&gas_handle));
		CUDA_SYNC_CHECK();

		CUDA_CHECK(cudaFree((void*)d_temp_buffer)); // << the UNcompacted, temporary output buffer
		CUDA_CHECK(cudaFree((void*)d_temp_buffer_output_gas));
		CUDA_CHECK(cudaFree((void*)compactedSizeBuffer));
	}
};

extern "C"
{   // TODO: rename texcoords to tex_positions
	__declspec(dllexport) Geometry* Geometry_Create(Context& context, const float* vertices, const float* texcoords, const int vertices_count)
	{
		return new Geometry(context, vertices, texcoords, vertices_count);
	}

	__declspec(dllexport) Geometry* IntIndexedGeometry_Create(
		Context& context,
		const float* vertices,
		const float* texcoords,
		const int vertices_count,
		const unsigned int* indices, 
		const int triangle_count)
	{
		return new Geometry(context, vertices, texcoords, vertices_count, indices, triangle_count);
	}

	__declspec(dllexport) Geometry* ShortIndexedGeometry_Create(
		Context& context,
		const float* vertices, 
		const float* texcoords,
		const int vertices_count,
		const unsigned short* indices, 
		const int triangle_count)
	{
		return new Geometry(context, vertices, texcoords, vertices_count, indices, triangle_count);
	}

	__declspec(dllexport) void Geometry_Destroy(Geometry* geometry) {
		delete geometry;
	}
}