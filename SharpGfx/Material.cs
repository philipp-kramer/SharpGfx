using System;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class Material : IDisposable
    {
        public virtual void Apply(Point3 cameraPosition) { }
        public virtual void UnApply() { }
        protected abstract void Dispose(bool disposing);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
