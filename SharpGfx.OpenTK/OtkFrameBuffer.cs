using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal class OtkFrameBuffer : FrameBuffer
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

        protected override void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
        }
    }
}