using System;
using System.Drawing;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal sealed class OtkFrameRenderBuffer : OtkFrameBuffer
    {
        private readonly OtkFrameBuffer _frameBuffer;
        private readonly int _renderHandle;

        public OtkFrameRenderBuffer(Size pixels)
        {
            _frameBuffer = new OtkFrameBuffer();

            _renderHandle = GL.GenRenderbuffer();
            GL.BindRenderbuffer(RenderbufferTarget.Renderbuffer, _renderHandle);
            GL.RenderbufferStorage(RenderbufferTarget.Renderbuffer, RenderbufferStorage.Depth24Stencil8, pixels.Width, pixels.Height);
            GL.FramebufferRenderbuffer(
                FramebufferTarget.Framebuffer,
                FramebufferAttachment.DepthStencilAttachment,
                RenderbufferTarget.Renderbuffer,
                _renderHandle);

            if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
            {
                throw new InvalidOperationException("framebuffer not configured correctly");
            }
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteRenderbuffer(_renderHandle);
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