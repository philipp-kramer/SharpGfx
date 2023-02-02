using System;

namespace SharpGfx.OpenGL;

internal class GlFrameBuffer : FrameBuffer, IDisposable
{
    private readonly GlApi _gl;
    internal readonly uint Handle;

    public GlFrameBuffer(GlApi gl)
    {
        _gl = gl;
        Handle = gl.GenFramebuffer();
        gl.BindFramebuffer(GlFramebufferTarget.Framebuffer, Handle);
    }

    private void ReleaseUnmanagedResources()
    {
        _gl.BindFramebuffer(GlFramebufferTarget.Framebuffer, Handle);
        _gl.DeleteFramebuffer(Handle);
        _gl.BindFramebuffer(GlFramebufferTarget.Framebuffer, 0);
    }

    protected virtual void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(true);
    }

    ~GlFrameBuffer()
    {
        _gl.Add(() => Dispose(false));
    }
}