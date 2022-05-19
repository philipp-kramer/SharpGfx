using System;

namespace SharpGfx.Primitives
{
    public interface Matrix3 : IPrimitive
    {
        public float this[int row, int col] { get; }

        public IVector3 Mul(IVector3 r);

        public static IVector3 operator *(Matrix3 l, IVector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }

    }
}
