using System;
using System.Collections;
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

        public void Dispose()
        {
            ReleaseUnmanagedResources();
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteBuffer(Handle);
        }

        ~OtkIndexBuffer()
        {
            UnmanagedRelease.Add(ReleaseUnmanagedResources);
        }
    }
}
