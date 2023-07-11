using System;

namespace SharpGfx.OptiX;

public readonly struct ContextPtr
{
    public readonly IntPtr Context;

    public ContextPtr(IntPtr context)
    {
        Context = context;
    }
}