using System;

namespace SharpGfx
{
    public abstract class Material
    {
        protected abstract void DoInContext(Action action);
        public abstract void Apply();
        public abstract void UnApply();
    }
}
