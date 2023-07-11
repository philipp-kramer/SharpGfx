#include "buffering.h"
#include "Exception.h"

size_t pixelFormatSize(BufferImageFormat format)
{
    switch (format)
    {
    case BufferImageFormat::UNSIGNED_BYTE4:
        return sizeof(char) * 4;
    case BufferImageFormat::FLOAT3:
        return sizeof(float) * 3;
    case BufferImageFormat::FLOAT4:
        return sizeof(float) * 4;
    default:
        throw std::invalid_argument("pixelFormatSize: Unrecognized buffer format");
    }
}

void ensureMinimumSize(int& w, int& h)
{
    if (w <= 0)
        w = 1;
    if (h <= 0)
        h = 1;
}