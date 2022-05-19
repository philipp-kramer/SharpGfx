using System;

namespace SharpGfx.Primitives
{
    public interface Matrix2 : IPrimitive
    {
        public float this[int row, int col] { get; }
        protected internal Array Values { get;  }
    }
}
