using System;

namespace SharpGfx
{
    public abstract class VertexBuffer : IDisposable
    {
        public abstract long Length { get; }

        protected abstract void Dispose(bool disposing);

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }
    }
}