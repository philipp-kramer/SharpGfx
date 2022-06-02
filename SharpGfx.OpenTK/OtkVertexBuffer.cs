using OpenTK.Graphics.OpenGL;

namespace SharpGfx.OpenTK
{
    internal class OtkVertexBuffer<T> : VertexBuffer
        where T : struct
    {
        internal readonly int Handle;

        public override long Length { get; }

        public OtkVertexBuffer(T[] data)
        {
            Length = data.Length;
            Handle = OtkBuffer.CreateBuffer(data, BufferTarget.ArrayBuffer);
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteBuffer(Handle);
        }

        protected override void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
        }

        ~OtkVertexBuffer()
        {
            UnmanagedRelease.Add(() => Dispose(false));
        }
    }
}