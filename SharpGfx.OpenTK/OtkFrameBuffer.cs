using System;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal class OtkFrameBuffer : FrameBuffer, IDisposable
    {
        internal readonly int Handle;

        public OtkFrameBuffer()
        {
            Handle = GL.GenFramebuffer();
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, Handle);
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteFramebuffer(Handle);
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
        }

        ~OtkFrameBuffer()
        {
            UnmanagedRelease.Add(() => Dispose(false));
        }
    }
}