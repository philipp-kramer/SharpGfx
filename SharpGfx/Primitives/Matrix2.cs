using System;

namespace SharpGfx.Primitives
{
    public interface Matrix2 : Primitive
    {
        public float this[int row, int col] { get; }
        protected internal Array Values { get;  }
    }
}
