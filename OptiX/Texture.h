#pragma once

#include <string>


struct CudaTexture
{
    cudaArray_t         array;
    cudaTextureObject_t object;
};

struct Texture {
public:
    uint32_t* pixel{ nullptr };
    int2 resolution{ -1 };

    Texture(uint32_t* pixels, const int width, const int height);
    Texture(const std::string& file_path);
    CudaTexture createCudaTexture();

    ~Texture()
    {
        if (pixel) delete[] pixel;
    }
};

extern "C"
{
    __declspec(dllexport) Texture* Texture_Create(uint32_t* pixels, const int width, const int height);
    __declspec(dllexport) void Texture_Destroy(Texture* texture);
}
