using System;

namespace SharpGfx.Primitives
{
    public interface Vector4 : IPrimitive
    {
        public float X { get; }
        public float Y { get; }
        public float Z { get; }
        public float W { get; }
        public float this[int index] { get; }
        public float Length { get; }
        public Array Values { get; }
        public IVector3 Xyz { get; }

        protected Vector4 Neg();
        protected Vector4 Add(Vector4 r);
        protected Vector4 Sub(Vector4 r);
        protected Vector4 Mul(float scalar);
        protected Vector4 Mul(Vector4 r);
        protected Vector4 Mul(Matrix4 r);
        public float Dot(Vector4 r);
        public Vector4 Normalized();

        public static Vector4 operator -(Vector4 v)
        {
            return v.Neg();
        }

        public static Vector4 operator +(Vector4 l, Vector4 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Add(r);
        }

        public static Vector4 operator -(Vector4 l, Vector4 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Sub(r);
        }

        public static Vector4 operator *(float scalar, Vector4 v)
        {
            return v.Mul(scalar);
        }

        public static Vector4 operator *(Vector4 v, float scalar)
        {
            return v.Mul(scalar);
        }

        public static Vector4 operator *(Vector4 l, Vector4 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }

        public static Vector4 operator *(Vector4 l, Matrix4 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }
    }
}
