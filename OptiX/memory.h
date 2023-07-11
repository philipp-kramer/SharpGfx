#pragma once
#include <cuda_runtime.h>
#include "Exception.h"

template<typename T> CUdeviceptr copy_array_to_device(const int count, const T* host_data)
{
    return copy_to_device(sizeof(T) * count, reinterpret_cast<const void*>(host_data));
}

CUdeviceptr copy_to_device(const size_t size, const void* host_data);
