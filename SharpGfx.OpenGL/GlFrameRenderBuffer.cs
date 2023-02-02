using System;

namespace SharpGfx.OpenGL;

internal sealed class GlFrameRenderBuffer : GlFrameBuffer
{
    private readonly GlApi _gl;
    private readonly uint _handle;

    public GlFrameRenderBuffer(GlApi gl, int width, int height, GlRenderbufferStorage storage, GlFramebufferAttachment attachment)
        : base(gl)
    {
        _gl = gl;

        _handle = gl.GenRenderbuffer();
        gl.BindRenderbuffer(GlRenderbufferTarget.Renderbuffer, _handle);
        gl.RenderbufferStorage(GlRenderbufferTarget.Renderbuffer, storage, width, height);
        gl.FramebufferRenderbuffer(GlFramebufferTarget.Framebuffer, attachment, GlRenderbufferTarget.Renderbuffer, _handle);

        var errorCode = gl.CheckFramebufferStatus(GlFramebufferTarget.Framebuffer);
        if (errorCode != GlFramebufferErrorCode.FramebufferComplete)
        {
            throw new InvalidOperationException($"framebuffer not configured correctly, error code {errorCode}");
        }
    }

    private void ReleaseUnmanagedResources()
    {
        _gl.DeleteRenderbuffer(_handle);
    }

    protected override void Dispose(bool disposing)
    {
        ReleaseUnmanagedResources();
        base.Dispose(disposing);
    }

    ~GlFrameRenderBuffer()
    {
        _gl.Add(() => Dispose(false));
    }
}