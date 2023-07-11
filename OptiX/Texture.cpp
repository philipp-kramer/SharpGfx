#include <map>
#include <stdint.h>
#include <string>
#include <vector_types.h>
#include <driver_types.h>
#include <cuda_runtime.h>

#include "Exception.h"
#include "Texture.h"
#include "stb_image.h"

Texture::Texture(uint32_t* pixels, const int width, const int height)
{
    pixel = pixels;
    resolution = int2();
    resolution.x = width;
    resolution.y = height;
}

Texture::Texture(const std::string& file_path)
{
    int2 res;
    int comp;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* image = stbi_load(file_path.c_str(), &res.x, &res.y, &comp, STBI_rgb_alpha);
    if (image) {
        resolution = res;
        pixel = (uint32_t*) image;
    }
    else
    {
        throw new std::exception(); // "Could not load texture from " + fileName + "!"
    }
}

CudaTexture Texture::createCudaTexture()
{
    CudaTexture cudaTexture{};
    cudaResourceDesc res_desc = {};

    cudaChannelFormatDesc channel_desc;
    int32_t width = resolution.x;
    int32_t height = resolution.y;
    int32_t pitch = width * sizeof(uint32_t);
    channel_desc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t& pixelArray = cudaTexture.array;

    CUDA_CHECK(cudaMallocArray(&pixelArray, &channel_desc, width, height));
    CUDA_CHECK(cudaMemcpy2DToArray(pixelArray, 0, 0, pixel, pitch, pitch, height, cudaMemcpyHostToDevice));

    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = pixelArray;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModePoint;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
    cudaTexture.object = cuda_tex;

    return cudaTexture;
}


Texture* Texture_Create(uint32_t* pixels, const int width, const int height) {
    return new Texture(pixels, width, height);
}

void Texture_Destroy(Texture* texture) {
    delete texture;
}
