using System;

namespace SharpGfx.OptiX;

public readonly struct GeometryPtr
{
    public readonly IntPtr Geometry;

    public GeometryPtr(IntPtr geometry)
    {
        Geometry = geometry;
    }
}