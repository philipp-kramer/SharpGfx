using System;

namespace SharpGfx.OpenGL;

internal sealed class GlIndexBuffer<T> : IDisposable
    where T : struct
{
    private readonly GlApi _gl;
    internal readonly uint Handle;
    internal readonly int Length;

    public GlIndexBuffer(GlApi gl, T[] data)
    {
        _gl = gl;
        Handle = GlBuffer.CreateBuffer(gl, data, GlBufferTarget.ElementArrayBuffer);
        Length = data.Length;
    }

    private void ReleaseUnmanagedResources()
    {
        _gl.DeleteBuffer(Handle);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        ReleaseUnmanagedResources();
    }

    ~GlIndexBuffer()
    {
        _gl.Add(ReleaseUnmanagedResources);
    }
}