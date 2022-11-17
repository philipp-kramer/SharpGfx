﻿using System;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL
{
    internal sealed class OglFrameRenderBuffer : OglFrameBuffer
    {
        private readonly OglFrameBuffer _frameBuffer;
        private readonly uint _handle;

        public OglFrameRenderBuffer(IVector2 pixels, GlRenderbufferStorage storage, GlFramebufferAttachment attachment)
        {
            _frameBuffer = new OglFrameBuffer();

            _handle = GL.GenRenderbuffer();
            GL.BindRenderbuffer(GlRenderbufferTarget.Renderbuffer, _handle);
            GL.RenderbufferStorage(GlRenderbufferTarget.Renderbuffer, storage, (int) pixels.X, (int) pixels.Y);
            GL.FramebufferRenderbuffer(
                GlFramebufferTarget.Framebuffer,
                attachment,
                GlRenderbufferTarget.Renderbuffer,
                _handle);

            if (GL.CheckFramebufferStatus(GlFramebufferTarget.Framebuffer) != GlFramebufferErrorCode.FramebufferComplete)
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