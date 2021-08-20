using System;

namespace SharpGfx.Primitives
{
    public interface Matrix4 : Primitive
    {
        public float this[int row, int col] { get; }
        protected internal Array Values { get; }

        public Matrix4 ToSpace(Space space);
        public Vector4 Mul(Vector4 r);
        public Matrix4 Mul(Matrix4 r);

        public static Matrix4 operator *(Matrix4 l, Matrix4 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }

        public static Vector4 operator *(Matrix4 l, Vector4 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }
    }
}
