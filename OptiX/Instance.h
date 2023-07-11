#pragma once
#include "Geometry.h"

extern "C"
{
	__declspec(dllexport) OptixInstance* Instance_Create(Geometry& geometry, float transform[12]) {
		OptixInstance* instance = new OptixInstance{};
		memcpy(instance->transform, transform, 12 * sizeof(float));
		instance->flags = OPTIX_INSTANCE_FLAG_NONE;
		instance->visibilityMask = 1u;
		instance->traversableHandle = geometry.gas_handle;

		return instance;
	}

	__declspec(dllexport) void Instance_Update(OptixInstance* instance, float transform[12]) {
		memcpy(instance->transform, transform, 12 * sizeof(float));
	}

	__declspec(dllexport) void Instance_Destroy(OptixInstance* instance) {
		delete instance;
	}
}