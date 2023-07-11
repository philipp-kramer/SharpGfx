using System;

namespace SharpGfx.OptiX;

public readonly struct WindowPtr
{
    public readonly IntPtr Window;

    public WindowPtr(IntPtr window)
    {
        Window = window;
    }
}