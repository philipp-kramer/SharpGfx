using System;

namespace SharpGfx.Primitives
{
    public interface Matrix3 : Primitive
    {
        public float this[int row, int col] { get; }

        public Vector3 Mul(Vector3 r);

        public static Vector3 operator *(Matrix3 l, Vector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }

    }
}
