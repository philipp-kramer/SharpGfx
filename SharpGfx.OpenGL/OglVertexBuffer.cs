namespace SharpGfx.OpenGL
{
    internal class OglVertexBuffer<T> : VertexBuffer
        where T : struct
    {
        internal readonly uint Handle;

        public override long Length { get; }

        public OglVertexBuffer(T[] data)
        {
            Length = data.Length;
            Handle = OglBuffer.CreateBuffer(data, GlBufferTarget.ArrayBuffer);
        }

        private void ReleaseUnmanagedResources()
        {
            GL.DeleteBuffer(Handle);
        }

        protected override void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
        }

        ~OglVertexBuffer()
        {
            Dispose(false);
        }
    }
}