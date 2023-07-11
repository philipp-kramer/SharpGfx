using System;

namespace SharpGfx.OptiX;

public readonly struct InstancePtr
{
    public readonly IntPtr Instance;

    public InstancePtr(IntPtr instance)
    {
        Instance = instance;
    }

    public bool IsZero => Instance == IntPtr.Zero;
}