using System;

namespace SharpGfx.OpenGL
{
    internal class OglFrameBuffer : FrameBuffer, IDisposable
    {
        internal readonly uint Handle;

        public OglFrameBuffer()
        {
            Handle = GL.GenFramebuffer();
            GL.BindFramebuffer(GlFramebufferTarget.Framebuffer, Handle);
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteFramebuffer(Handle);
            GL.BindFramebuffer(GlFramebufferTarget.Framebuffer, 0);
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

        ~OglFrameBuffer()
        {
            Dispose(false);
        }
    }
}