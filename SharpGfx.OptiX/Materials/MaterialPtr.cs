using System;

namespace SharpGfx.OptiX.Materials;

public readonly struct MaterialPtr
{
    public readonly IntPtr Material;

    public MaterialPtr(IntPtr material)
    {
        Material = material;
    }
}