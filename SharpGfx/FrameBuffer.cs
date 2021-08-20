using System;

namespace SharpGfx
{
    public abstract class FrameBuffer : IDisposable
    {
        protected abstract void Dispose(bool disposing);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        ~FrameBuffer()
        {
            Dispose(false);
        }
    }
}