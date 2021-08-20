using System;

namespace SharpGfx.Primitives
{
    public interface Vector3 : Primitive
    {
        public float X { get; }
        public float Y { get; }
        public float Z { get; }

        public float this[int index] { get; }
        public Array Values { get; }
        public float Length { get; }
        public Vector2 Xy { get; }
        public Vector2 Xz { get; }
        public Vector2 Yz { get; }

        public Vector4 Extend(float w);
        protected Vector3 Neg();
        protected Vector3 Add(Vector3 r);
        protected Vector3 Sub(Vector3 r);
        protected Vector3 Mul(float scalar);
        protected Vector3 Mul(Vector3 r);
        public Vector3 Cross(Vector3 r);
        public float Dot(Vector3 r);
        public Vector3 Normalized();

        public static Vector3 operator +(Vector3 l, Vector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Add(r);
        }

        public static Vector3 operator -(Vector3 v)
        {
            return v.Neg();
        }

        public static Vector3 operator -(Vector3 l, Vector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Sub(r);
        }

        public static Vector3 operator *(float scalar, Vector3 v)
        {
            return v.Mul(scalar);
        }

        public static Vector3 operator *(Vector3 v, float scalar)
        {
            return v.Mul(scalar);
        }

        public static Vector3 operator *(Vector3 l, Vector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Mul(r);
        }

        public static Vector3 Cross(Vector3 l, Vector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Cross(r);
        }

        public static float Dot(Vector3 l, Vector3 r)
        {
            if (!IsVisible(l, r)) throw new ArgumentException("cross space operation");
            return l.Dot(r);
        }
    }
}
