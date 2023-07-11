using System;

namespace SharpGfx.OptiX;

public readonly struct CameraPtr
{
    public readonly IntPtr Camera;

    public CameraPtr(IntPtr camera)
    {
        Camera = camera;
    }
}