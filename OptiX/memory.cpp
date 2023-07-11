#include <cuda_runtime.h>
#include "Exception.h"
#include "memory.h"

CUdeviceptr copy_to_device(const size_t size, const void* host_data)
{
    try
    {
        CUdeviceptr device_data = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_data), size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(device_data), host_data, size, cudaMemcpyHostToDevice));
        return device_data;
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return NULL;
    }
}