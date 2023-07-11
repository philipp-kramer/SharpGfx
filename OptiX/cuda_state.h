#pragma once

struct CudaState
{
	OptixDeviceContext          context = 0;
	CUstream                    stream = 0;
	OptixPipeline               pipeline = 0;
	OptixShaderBindingTable     sbt = {};
	CUDAOutputBuffer<uchar4>* output_buffer = nullptr;

	CUdeviceptr dOptixInstances;
	OptixBuildInput instanceBuildInput{};
	OptixAccelBuildOptions acceleratorBuildOptions{};
	OptixAccelBufferSizes acceleratorBufferSizes;
	CUdeviceptr dInstanceAcceleratorBuffer = 0;
};
