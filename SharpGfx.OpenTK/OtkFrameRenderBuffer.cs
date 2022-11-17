using System;
using OpenTK.Graphics.OpenGL;
using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    internal sealed class OtkFrameRenderBuffer : OtkFrameBuffer
    {
        private readonly OtkFrameBuffer _frameBuffer;
        private readonly int _handle;

        public OtkFrameRenderBuffer(IVector2 pixels, RenderbufferStorage storage, FramebufferAttachment attachment)
        {
            _frameBuffer = new OtkFrameBuffer();

            _handle = GL.GenRenderbuffer();
            GL.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _handle);
            GL.RenderbufferStorage(RenderbufferTarget.Renderbuffer, storage, (int) pixels.X, (int) pixels.Y);
            GL.FramebufferRenderbuffer(
                FramebufferTarget.Framebuffer,
                attachment,
                RenderbufferTarget.Renderbuffer,
                _handle);

            if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
            {
                throw new InvalidOperationException("framebuffer not configured correctly");
            }
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteRenderbuffer(_handle);
        }

        protected override void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
            if (disposing)
            {
                _frameBuffer.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}