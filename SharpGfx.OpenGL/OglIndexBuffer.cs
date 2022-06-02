using System;

namespace SharpGfx.OpenGL
{
    internal sealed class OglIndexBuffer<T> : IDisposable
        where T : struct
    {
        internal readonly uint Handle;
        internal readonly int Length;

        public OglIndexBuffer(T[] data)
        {
            Handle = OglBuffer.CreateBuffer(data, GlBufferTarget.ElementArrayBuffer);
            Length = data.Length;
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteBuffer(Handle);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            ReleaseUnmanagedResources();
        }

        ~OglIndexBuffer()
        {
            ReleaseUnmanagedResources();
        }
    }
}
