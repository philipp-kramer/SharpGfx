#pragma once
#include <glad/glad.h>

enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data = nullptr;
    unsigned int      width = 0;
    unsigned int      height = 0;
    BufferImageFormat pixel_format;
};

void ensureMinimumSize(int& w, int& h);
size_t pixelFormatSize(BufferImageFormat format);
