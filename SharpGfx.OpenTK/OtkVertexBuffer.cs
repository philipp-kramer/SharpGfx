using System;
using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal class OtkVertexBuffer<T> : VertexBuffer, IDisposable
        where T : struct
    {
        internal readonly int Handle;

        public override long Length { get; }

        public OtkVertexBuffer(T[] data)
        {
            Length = data.Length;
            Handle = OtkBuffer.CreateBuffer(data, BufferTarget.ArrayBuffer);
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteBuffer(Handle);
        }

        ~OtkVertexBuffer()
        {
            UnmanagedRelease.Add(() => Dispose(false));
        }
    }
}