using System;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal sealed class OtkIndexBuffer<T> : IDisposable
        where T : struct
    {
        internal readonly int Handle;
        internal readonly int Length;

        public OtkIndexBuffer(T[] data)
        {
            Handle = OtkBuffer.CreateBuffer(data, BufferTarget.ElementArrayBuffer);
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

        ~OtkIndexBuffer()
        {
            UnmanagedRelease.Add(ReleaseUnmanagedResources);
        }
    }
}
